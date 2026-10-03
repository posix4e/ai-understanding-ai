"""Render one shared manuscript JSON to a research PDF and standalone LaTeX.

Usage: work/paper-venv/bin/python paper/build.py --input outputs/paper/manuscript.json
       --output-dir outputs/paper
No research environment, model, checkpoint, or scientific result is imported.
Synthetic tests must use --output-dir work/renderer_demo.
"""

import argparse
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/paper-matplotlib-cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, mathtext
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                               PageBreak, PageTemplate, Paragraph, Spacer, Table,
                               TableStyle, XPreformatted)
from pypdf import PdfReader


COLORS = ["#2563a6", "#c66a24", "#328276", "#9961a0", "#637082", "#ad4747"]
WIDTH, HEIGHT = 612, 792
LEFT, RIGHT, TOP, BOTTOM = 58, 58, 57, 51
CONTENT = WIDTH-LEFT-RIGHT


def normalize(text):
    """Use shared Unicode punctuation normalization for both target formats."""
    return str(text).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", " - ").replace("\u2212", "-")


def tex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    text = normalize(text)
    # Greek/comparison symbols remain usable under conventional pdfLaTeX.
    unicode_tex = {"≥": r"\ensuremath{\geq}", "≤": r"\ensuremath{\leq}", "×": r"\ensuremath{\times}",
                   "±": r"\ensuremath{\pm}", "→": r"\ensuremath{\rightarrow}", "←": r"\ensuremath{\leftarrow}",
                   "α": r"\ensuremath{\alpha}", "β": r"\ensuremath{\beta}", "Δ": r"\ensuremath{\Delta}",
                   "μ": r"\ensuremath{\mu}", "π": r"\ensuremath{\pi}", "∈": r"\ensuremath{\in}", "∞": r"\ensuremath{\infty}"}
    return "".join(unicode_tex.get(c, replacements.get(c, c)) for c in text)


def xml(text):
    return html.escape(normalize(text), quote=False).replace("\n", "<br/>")


def paragraph_text(block):
    return xml(block["text"])


def math_source(block):
    source = block["math"].strip()
    return source[1:-1] if source.startswith("$") and source.endswith("$") else source


def horizontal_lines(block):
    lines = block.get("hline", [])
    if not isinstance(lines, list):
        lines = [lines]
    return [line if isinstance(line, dict) else {"value": line} for line in lines]


def validate(manuscript):
    for field in ("title", "abstract", "sections", "references"):
        if field not in manuscript:
            raise ValueError(f"Missing manuscript field: {field}")
    figure_ids = set()
    for section in manuscript["sections"]:
        if not isinstance(section.get("title"), str):
            raise ValueError("Each section requires a title")
        for block in section["blocks"]:
            kind = block["type"]
            if kind not in ("paragraph", "equation", "table", "figure", "code", "pagebreak"):
                raise ValueError(f"Unsupported block type {kind}")
            if kind == "table":
                if not block["headers"] or any(len(row) != len(block["headers"]) for row in block["rows"]):
                    raise ValueError("Table cells must match a nonempty header")
            if kind != "figure":
                continue
            name = block["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in figure_ids:
                raise ValueError("Figure IDs must be unique filesystem-safe names")
            figure_ids.add(name)
            if block["kind"] not in ("errorbar", "grouped_bar") or not block["series"]:
                raise ValueError("Expected nonempty errorbar or grouped_bar figure")
            n = len(block["xlabels"])
            for series in block["series"]:
                values = np.asarray(series["values"], dtype=float)
                if values.shape != (n,) or not np.isfinite(values).all():
                    raise ValueError(f"{name}: invalid figure values")
                if ("lower" in series) != ("upper" in series):
                    raise ValueError("Both confidence interval bounds are required")
                if "lower" in series:
                    lower, upper = np.array(series["lower"]), np.array(series["upper"])
                    if lower.shape != (n,) or upper.shape != (n,) or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > values) or np.any(upper < values):
                        raise ValueError("Interval bounds must be finite and contain their plotted values")
            for line in horizontal_lines(block):
                if not math.isfinite(float(line["value"])):
                    raise ValueError("Nonfinite horizontal reference")


def chart(block, folder):
    """Create standard data-only matplotlib PNG/SVG, never a PDF figure."""
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(6.7, float(block.get("height_inches", 2.85))))
    x = np.arange(len(block["xlabels"]))
    count = len(block["series"])
    width = .72/count
    for index, series in enumerate(block["series"]):
        values = np.array(series["values"], dtype=float)
        offset = (index-(count-1)/2)*width
        error = None
        if "lower" in series:
            error = np.array([values-np.array(series["lower"]), np.array(series["upper"])-values])
        if block["kind"] == "grouped_bar":
            ax.bar(x+offset, values, width=width*.92, yerr=error, capsize=2.5,
                   color=COLORS[index % len(COLORS)], label=normalize(series["label"]), linewidth=.5)
        else:
            ax.errorbar(x+offset, values, yerr=error, fmt="o", markersize=4,
                        elinewidth=1, capsize=2.5, color=COLORS[index % len(COLORS)], label=normalize(series["label"]))
    for line in horizontal_lines(block):
        ax.axhline(float(line["value"]), color="#777777", linewidth=.8, linestyle="--", label=normalize(line["label"]) if line.get("label") else None)
    ax.set_xticks(x, [normalize(v) for v in block["xlabels"]])
    ax.set_ylabel(normalize(block.get("ylabel", "")))
    ax.set_xlabel(normalize(block.get("xlabel", "")))
    if block.get("title"):
        ax.set_title(normalize(block["title"]), fontsize=10, pad=9)
    if block.get("ylim") is not None:
        ax.set_ylim(*block["ylim"])
    ax.grid(axis="y", color="#e1e5e8", linewidth=.6)
    ax.set_axisbelow(True)
    if count > 1 or any(line.get("label") for line in horizontal_lines(block)):
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.38), frameon=False,
                  ncol=min(count+1, 3), fontsize=8)
    fig.tight_layout()
    paths = {suffix: folder/f"{block['id']}.{suffix}" for suffix in ("png", "svg")}
    for suffix, path in paths.items():
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.06, format=suffix)
    plt.close(fig)
    return paths


def equation_image(source, folder, number):
    path = folder/f"equation_{number:02d}.png"
    mathtext.math_to_image("$"+source+"$", path, dpi=300, format="png",
                         prop=font_manager.FontProperties(size=12), color="#17212b")
    return path


class ResearchDocument(BaseDocTemplate):
    def __init__(self, path, manuscript):
        super().__init__(str(path), pagesize=(WIDTH, HEIGHT), leftMargin=LEFT, rightMargin=RIGHT,
                         topMargin=TOP, bottomMargin=BOTTOM, title=normalize(manuscript["title"]),
                         author=normalize(manuscript.get("author", "")))
        self.manuscript = manuscript
        self.addPageTemplates(PageTemplate(id="research", frames=[Frame(LEFT, BOTTOM, CONTENT, HEIGHT-TOP-BOTTOM,
                                                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#5a626b"))
        canvas.setFont("ResearchSerif", 7.5)
        if doc.page > 1:
            title = normalize(self.manuscript.get("short_title", self.manuscript["title"]))
            while pdfmetrics.stringWidth(title, "ResearchSerif", 7.5) > CONTENT-55:
                title = title[:-5].rstrip()+"..."
            canvas.drawString(LEFT, HEIGHT-31, title)
            canvas.setStrokeColor(colors.HexColor("#d6dbe0"))
            canvas.line(LEFT, HEIGHT-39, WIDTH-RIGHT, HEIGHT-39)
        canvas.drawString(LEFT, 29, normalize(self.manuscript.get("status", "")))
        canvas.drawRightString(WIDTH-RIGHT, 29, str(doc.page))
        canvas.restoreState()


def image_flowable(path, max_width=CONTENT, max_height=230):
    width, height = ImageReader(str(path)).getSize()
    scale = min(max_width/width, max_height/height)
    flow = Image(str(path), width=width*scale, height=height*scale)
    flow.hAlign = "CENTER"
    return flow


def render_pdf(manuscript, folder, charts):
    font_folder = Path(matplotlib.get_data_path())/"fonts/ttf"
    for name, filename in [("ResearchSerif", "DejaVuSerif.ttf"), ("ResearchSerifBold", "DejaVuSerif-Bold.ttf"),
                           ("ResearchSerifItalic", "DejaVuSerif-Italic.ttf"), ("ResearchMono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_folder/filename)))
    pdfmetrics.registerFontFamily("ResearchSerif", normal="ResearchSerif", bold="ResearchSerifBold", italic="ResearchSerifItalic")
    body = ParagraphStyle("body", fontName="ResearchSerif", fontSize=10.5, leading=14.1, alignment=TA_JUSTIFY, spaceAfter=7)
    title = ParagraphStyle("title", parent=body, fontName="ResearchSerifBold", fontSize=19, leading=23, alignment=TA_LEFT, spaceAfter=10)
    subtitle = ParagraphStyle("subtitle", parent=body, fontSize=10, leading=14, textColor=colors.HexColor("#53606c"), alignment=TA_LEFT, spaceAfter=9)
    heading = ParagraphStyle("heading", parent=body, fontName="ResearchSerifBold", fontSize=12.3, leading=16, spaceBefore=14, spaceAfter=7, keepWithNext=True, alignment=TA_LEFT)
    caption = ParagraphStyle("caption", parent=body, fontSize=8.6, leading=11.5, alignment=TA_LEFT, spaceAfter=10)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10.3, alignment=TA_LEFT, spaceAfter=0)
    abstract = ParagraphStyle("abstract", parent=body, fontSize=9.7, leading=13, leftIndent=12, rightIndent=12, spaceAfter=12)
    code = ParagraphStyle("code", fontName="ResearchMono", fontSize=7.3, leading=10, leftIndent=7, rightIndent=7, spaceBefore=4, spaceAfter=9)
    story = [Paragraph(xml(manuscript["title"]), title)]
    for text in (manuscript.get("subtitle"), manuscript.get("author"), manuscript.get("date")):
        if text:
            story.append(Paragraph(xml(text), subtitle))
    story += [Paragraph("Abstract", heading), Paragraph(xml(manuscript["abstract"]), abstract)]
    figures = tables = equations = 0
    for number, section in enumerate(manuscript["sections"], 1):
        story.append(Paragraph(f"{number}. {xml(section['title'])}", heading))
        for block in section["blocks"]:
            kind = block["type"]
            if kind == "paragraph":
                story.append(Paragraph(paragraph_text(block), body))
            elif kind == "equation":
                equations += 1
                path = equation_image(math_source(block), folder/"figures", equations)
                pixel_width, pixel_height = ImageReader(str(path)).getSize()
                scale = min(72/300, (CONTENT-38)/pixel_width)
                flow = Image(str(path), width=pixel_width*scale, height=pixel_height*scale)
                display = Table([[flow, Paragraph(f"({equations})", caption)]], colWidths=[CONTENT-30, 30])
                display.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (0, 0), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
                story += [Spacer(1, 3), display, Spacer(1, 7)]
            elif kind == "figure":
                figures += 1
                story.append(KeepTogether([image_flowable(charts[block["id"]]["png"], max_height=250), Spacer(1, 5),
                                          Paragraph(f"<b>Figure {figures}.</b> {xml(block.get('caption', ''))}", caption)]))
            elif kind == "table":
                tables += 1
                rows = [[Paragraph(f"<b>{xml(v)}</b>", cell) for v in block["headers"]]]
                rows += [[Paragraph(xml(v), cell) for v in row] for row in block["rows"]]
                widths = block.get("widths")
                widths = [CONTENT*w/sum(widths) for w in widths] if widths else [CONTENT/len(block["headers"])]*len(block["headers"])
                tab = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
                tab.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), .8, colors.HexColor("#425363")),
                    ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#7f8e99")), ("LINEBELOW", (0, -1), (-1, -1), .6, colors.HexColor("#7f8e99")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                cap = Paragraph(f"<b>Table {tables}.</b> {xml(block.get('caption', ''))}", caption)
                cap.keepWithNext = True
                story += [Spacer(1, 4), cap, tab, Spacer(1, 10)]
            elif kind == "code":
                story.append(XPreformatted(xml(block["text"]), code, maxLineLength=92, splitChars=" /_-", newLineChars=""))
            elif kind == "pagebreak":
                story.append(PageBreak())
    story.append(Paragraph("References", heading))
    for ref in manuscript["references"]:
        url = ref.get("url")
        link = f' <link href="{html.escape(url, quote=True)}" color="#2563a6">Source</link>.' if url else ""
        story.append(Paragraph(f"[{xml(ref['id'])}] {xml(ref['text'])}{link}", caption))
    path = folder/"paper.pdf"
    ResearchDocument(path, manuscript).build(story)
    return path


def tex_figure(block, number):
    """Render plots entirely as inline PGFPlots coordinates: no external assets."""
    labels = ",".join("{"+tex_escape(x)+"}" for x in block["xlabels"])
    options = [r"width=0.95\linewidth", "height=5.6cm", "grid=major", "major grid style={gray!20}",
               "axis lines=left", "tick label style={font=\\small}", "label style={font=\\small}",
               "xtick={"+",".join(map(str, range(len(block["xlabels"]))))+"}", "xticklabels={"+labels+"}",
               "ylabel={"+tex_escape(block.get("ylabel", ""))+"}", "title={"+tex_escape(block.get("title", ""))+"}",
               "xlabel={"+tex_escape(block.get("xlabel", ""))+"}",
               r"legend style={at={(0.5,-0.23)},anchor=north,draw=none,font=\small}", "legend columns=2"]
    if block.get("ylim"):
        options.extend([f"ymin={block['ylim'][0]}", f"ymax={block['ylim'][1]}"])
    if block["kind"] == "grouped_bar":
        options.append("ybar")
        options.append("bar width=7pt")
    out = [r"\begin{figure}[htbp]", r"\centering", r"\begin{tikzpicture}", "\\begin{axis}["+",\n".join(options)+"]"]
    for index, series in enumerate(block["series"]):
        color = f"papercolor{index % len(COLORS)}"
        style = [f"color={color}", f"fill={color}"]
        if block["kind"] == "errorbar":
            style += ["only marks", "mark=*", "mark size=1.8pt"]
        if "lower" in series:
            style += ["error bars/.cd", "y dir=both", "y explicit", "error bar style={line width=.6pt}"]
        out.append("\\addplot+["+",".join(style)+"] coordinates {")
        for point_index, value in enumerate(series["values"]):
            x = point_index
            if block["kind"] == "errorbar":
                x += (index-(len(block["series"])-1)/2)*(.72/len(block["series"]))
            if "lower" in series:
                out.append(f"({x},{value}) += (0,{series['upper'][point_index]-value}) -= (0,{value-series['lower'][point_index]})")
            else:
                out.append(f"({x},{value})")
        out.extend(["};", "\\addlegendentry{"+tex_escape(series["label"])+"}"])
    for line in horizontal_lines(block):
        legend = "" if line.get("label") else ",forget plot"
        out.append(f"\\addplot[gray,dashed,domain=-0.45:{len(block['xlabels'])-.55},samples=2{legend}] " + "{" + str(line["value"]) + "};")
        if line.get("label"):
            out.append("\\addlegendentry{"+tex_escape(line["label"])+"}")
    out += [r"\end{axis}", r"\end{tikzpicture}", "\\caption{"+tex_escape(block.get("caption", ""))+"}",
            "\\label{fig:"+block["id"]+"}", r"\end{figure}"]
    return "\n".join(out)


def render_tex(manuscript, folder):
    out = [r"\documentclass[10pt,letterpaper]{article}", r"\usepackage[margin=0.8in]{geometry}",
           r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}", r"\usepackage{lmodern}",
           r"\usepackage{amsmath,amssymb,booktabs,array,longtable}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}",
           r"\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}", r"\usepackage{fancyhdr}",
           r"\setlength{\parindent}{0pt}", r"\setlength{\parskip}{5pt}", r"\setlength{\headheight}{14pt}",
           r"\pagestyle{fancy}\fancyhf{}", r"\fancyfoot[R]{\thepage}",
           "\\fancyfoot[L]{\\scriptsize "+tex_escape(manuscript.get("status", ""))+"}",
           "\\fancyhead[L]{\\scriptsize "+tex_escape(manuscript.get("short_title", manuscript["title"]))+"}",
           r"\renewcommand{\headrulewidth}{0.2pt}"]
    out += [f"\\definecolor{{papercolor{i}}}{{HTML}}{{{color[1:]}}}" for i, color in enumerate(COLORS)]
    out += ["\\title{"+tex_escape(manuscript["title"])+"}", "\\author{"+tex_escape(manuscript.get("author", ""))+"}",
            "\\date{"+tex_escape(manuscript.get("date", ""))+"}", r"\begin{document}",
            r"\fontsize{10.5}{14.1}\selectfont", r"\maketitle"]
    if manuscript.get("subtitle"):
        out += [r"\begin{center}", tex_escape(manuscript["subtitle"]), r"\end{center}"]
    out += [r"\begin{abstract}", tex_escape(manuscript["abstract"]), r"\end{abstract}"]
    figures = tables = 0
    for section in manuscript["sections"]:
        out.append("\\section{"+tex_escape(section["title"])+"}")
        for block in section["blocks"]:
            kind = block["type"]
            if kind == "paragraph":
                out += [block.get("latex", tex_escape(block["text"])), ""]
            elif kind == "equation":
                out += [r"\begin{equation}", math_source(block), r"\end{equation}"]
            elif kind == "figure":
                figures += 1
                out.append(tex_figure(block, figures))
            elif kind == "table":
                tables += 1
                n = len(block["headers"])
                weights = block.get("widths", [1]*n)
                columns = "".join(">{\\raggedright\\arraybackslash}p{\\dimexpr "+f"{w/sum(weights):.5f}"+"\\linewidth-2\\tabcolsep\\relax}" for w in weights)
                out += ["\\begin{longtable}{@{}"+columns+"@{}}", "\\caption{"+tex_escape(block.get("caption", ""))+r"}\\",
                        r"\toprule", " & ".join("\\textbf{"+tex_escape(v)+"}" for v in block["headers"])+r"\\", r"\midrule\endhead"]
                out += [" & ".join(tex_escape(v) for v in row)+r"\\" for row in block["rows"]]
                out += [r"\bottomrule", r"\end{longtable}"]
            elif kind == "code":
                if "\\end{verbatim}" in block["text"]:
                    raise ValueError("Code cannot contain a literal end-verbatim delimiter")
                out += [r"{\small\begin{verbatim}", normalize(block["text"]), r"\end{verbatim}}"]
            elif kind == "pagebreak":
                out.append(r"\clearpage")
    out += [r"\begin{thebibliography}{99}"]
    for ref in manuscript["references"]:
        name = str(ref["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Reference IDs must use letters, digits, underscore or hyphen")
        out.append("\\bibitem["+tex_escape(name)+"]{"+name+"} "+tex_escape(ref["text"]))
        if ref.get("url"):
            out.append("\\url{"+ref["url"].replace("%", r"\%")+"}.")
    out += [r"\end{thebibliography}", r"\end{document}", ""]
    path = folder/"paper.tex"
    path.write_text("\n".join(out))
    return path


def build(input_path, folder):
    source = input_path.read_bytes()
    manuscript = json.loads(source)
    validate(manuscript)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"figures").mkdir(exist_ok=True)
    charts = {block["id"]: chart(block, folder/"figures") for section in manuscript["sections"] for block in section["blocks"] if block["type"] == "figure"}
    pdf = render_pdf(manuscript, folder, charts)
    tex = render_tex(manuscript, folder)
    reader = PdfReader(pdf)
    metadata = {"input": str(input_path), "input_sha256": hashlib.sha256(source).hexdigest(), "pages": len(reader.pages),
                "pdf": str(pdf), "tex": str(tex), "figures": {name: {kind: str(path) for kind, path in files.items()} for name, files in charts.items()},
                "shared_content": "Both renderers consume the same JSON; paragraph latex overrides are formatting supplied by the author.",
                "pdf_text_characters": sum(len(page.extract_text() or "") for page in reader.pages)}
    (folder/"build_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/paper/manuscript.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper"))
    args = parser.parse_args()
    build(args.input, args.output_dir)


if __name__ == "__main__":
    main()
