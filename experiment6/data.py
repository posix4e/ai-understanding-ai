"""Fixed GPT-2 lookup formats and exact token-occurrence serialization."""
from dataclasses import dataclass
import random

import torch

KEYS = ["cat", "dog", "bird", "fish", "tree", "book", "door", "road",
        "king", "girl", "boy", "man", "car", "ball", "hand", "head"]
VALUES = list(range(20, 36))
TEMPLATES = ["table", "repeat", "one_demo", "two_demo"]
DEMO1 = "Table: red = 6; blue = 2; green = 9; yellow = 4;\nLookup: blue = 2\n\n"
DEMO2 = "Table: sun = 8; moon = 3; star = 5; cloud = 1;\nLookup: star = 5\n\n"


def identity(case):
    """Exclude the same association map in every order and query position."""
    return tuple(sorted(zip(case["keys"], case["values"])))


def generate_cases(n, seed, excluded=()):
    if n % 4:
        raise ValueError("balanced sample size must be divisible by four")
    rng = random.Random(seed)
    forbidden = {identity(c) for c in excluded}
    cases = []
    queries = list(range(4)) * (n // 4)
    rng.shuffle(queries)
    for query in queries:
        while True:
            case = {"keys": rng.sample(KEYS, 4), "values": rng.sample(VALUES, 4),
                    "query_pair": query}
            marker = identity(case)
            if marker not in forbidden:
                forbidden.add(marker)
                cases.append(case)
                break
    return cases


@dataclass
class EncodedCase:
    ids: torch.Tensor
    chunks: list[list[int]]
    suffix: list[int]
    prefix: list[int]
    text: str

    def permutation(self, grouped=False):
        order = [0, 2, 4, 6, 1, 3, 5, 7] if grouped else list(range(8))
        indices = self.prefix + [t for c in order for t in self.chunks[c]] + self.suffix
        assert sorted(indices) == list(range(self.ids.numel()))
        return torch.tensor(indices, dtype=torch.long)


def encode_case(tokenizer, case, template):
    if template not in TEMPLATES:
        raise ValueError(template)
    prefix = {"table": "Table:", "repeat": "Values:",
              "one_demo": DEMO1 + "Table:",
              "two_demo": DEMO1 + DEMO2 + "Table:"}[template]
    suffix = ("\nValues:" if template == "repeat" else "\nLookup:")
    suffix += " " + case["keys"][case["query_pair"]] + " ="
    pieces = [prefix]
    for key, value in zip(case["keys"], case["values"]):
        pieces += [f" {key} =", f" {value};"]
    pieces += [suffix]
    spans, offset = [], 0
    for piece in pieces:
        spans.append((offset, offset + len(piece)))
        offset += len(piece)
    text = "".join(pieces)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    ownership = [[] for _ in pieces]
    for t, (a, b) in enumerate(encoded["offset_mapping"]):
        owners = [i for i, (left, right) in enumerate(spans) if left <= a < b <= right]
        if len(owners) != 1:
            raise ValueError(f"token crosses chunk boundary: {t}, {(a,b)}")
        ownership[owners[0]].append(t)
    if any(not x for x in ownership):
        raise ValueError("empty token chunk")
    if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != text:
        raise ValueError("native tokenization does not roundtrip")
    if len({len(ownership[i]) for i in (1, 3, 5, 7)}) != 1:
        raise ValueError("key chunks have different token lengths")
    answers = answer_ids(tokenizer)
    target = case["values"][case["query_pair"]]
    if tokenizer.encode(text + f" {target}", add_special_tokens=False) != ids + [answers[VALUES.index(target)]]:
        raise ValueError("answer token changes in the actual suffix context")
    return EncodedCase(torch.tensor(ids, dtype=torch.long), ownership[1:9],
                       ownership[-1], ownership[0], text)


def answer_ids(tokenizer):
    result = []
    for v in VALUES:
        ids = tokenizer.encode(f" {v}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"answer is not a single token: {v}")
        result += ids
    if len(set(result)) != len(VALUES):
        raise ValueError("nonunique answer tokens")
    return result
