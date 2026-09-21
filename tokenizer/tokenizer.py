# JAI Version: 0.1.0
"""A transparent, deterministic tokenizer built from scratch for JAI."""
import json
import re
from pathlib import Path

class Tokenizer:
    """Convert text to token IDs and back using a learned vocabulary."""
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    _TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        special = [self.PAD, self.UNK, self.BOS, self.EOS]
        vocabulary = vocabulary or []
        self.tokens = []
        for token in special + vocabulary:
            if token not in self.tokens: self.tokens.append(token)
        self._rebuild_maps()

    def _rebuild_maps(self) -> None:
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for index, token in enumerate(self.tokens)}

    @classmethod
    def split(cls, text: str) -> list[str]:
        if not isinstance(text, str): raise TypeError("text must be a string.")
        return cls._TOKEN_PATTERN.findall(text)

    def build_vocabulary(self, texts: list[str], min_frequency: int = 1) -> None:
        if min_frequency < 1: raise ValueError("min_frequency must be positive.")
        counts: dict[str, int] = {}
        for text in texts:
            for token in self.split(text): counts[token] = counts.get(token, 0) + 1
        new_tokens = sorted(token for token, count in counts.items() if count >= min_frequency and token not in self.token_to_id)
        self.tokens.extend(new_tokens); self._rebuild_maps()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = []
        if add_bos: ids.append(self.token_to_id[self.BOS])
        ids.extend(self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in self.split(text))
        if add_eos: ids.append(self.token_to_id[self.EOS])
        return ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        special = {self.PAD, self.UNK, self.BOS, self.EOS}; pieces = []
        for token_id in ids:
            if token_id not in self.id_to_token: raise ValueError(f"Unknown token ID: {token_id}.")
            token = self.id_to_token[token_id]
            if skip_special and token in special: continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        data = {"version": "0.1.0", "tokens": self.tokens}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != "0.1.0": raise ValueError("Unsupported tokenizer vocabulary version.")
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens): raise ValueError("Invalid tokenizer vocabulary.")
        return cls(tokens)
