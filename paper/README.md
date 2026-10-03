# Research manuscript and independent audit

The scientific source files and results are described in
[E5 reproduction instructions](../outputs/experiment5/REPRODUCE.md).
These manuscript tools are later presentation and audit code; they are **not**
part of the preregistered experimental implementation.

## Build the paper

From the repository root, preserve published manuscript outputs if comparing
versions. Create a separate presentation environment, leaving `.venv` unchanged:

```sh
uv venv work/paper-venv --python python3.14
uv pip sync --python work/paper-venv/bin/python paper/requirements.txt
.venv/bin/python paper/assemble.py
work/paper-venv/bin/python paper/build.py --input outputs/paper/manuscript.json --output-dir outputs/paper
```

`assemble.py` reads the frozen result tables and writes the shared manuscript
JSON. It does not evaluate models or alter scientific outputs. `build.py`
renders `paper.pdf` with ReportLab, standalone `paper.tex` with inline PGFPlots
coordinates, and three figures as PNG/SVG. The JSON hash is recorded in
`build_metadata.json`. The PDF and LaTeX share text and data; typography and page
breaks can differ. The LaTeX source has no external asset dependencies and can
be opened in the Codex editor or compiled with a normal LaTeX installation.

Human authorship, affiliations, disclosures, venue format, and submission remain
the responsibility of the researchers. The project name is a document label,
not an assertion that an AI agent qualifies as an author. The manuscript is not
peer reviewed.

## Repeat the independent outcome audit

`audit_e5.py` is the exact independent analysis used for the skeptical review.
It imports neither the model nor the official scorer and runs no forwards.
Run in a separate checkout or preserve the published audit JSON before running:

```sh
mkdir -p work
for seed in 6 7 8 9 10 11; do
  .venv/bin/python paper/audit_e5.py --seed "$seed"
done
.venv/bin/python paper/audit_e5.py --final
```

Per-seed intermediate audits are written under `work/`; `--final` validates
the frozen local file hashes, checks timing, compares independently computed
values with the official scorer, and writes
`outputs/experiment5/independent_audit.json`. It does not reproduce the original
remote verification event. State-error arrays remain measurements from the
registered runner; its random-model tests audit their implementation.
