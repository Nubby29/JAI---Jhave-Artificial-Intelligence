# JAI Version: 0.1.0
"""Tests for JAI tokenizer."""
import tempfile
import unittest
from pathlib import Path
from tokenizer.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    def test_split_preserves_whitespace_and_punctuation(self) -> None:
        self.assertEqual(Tokenizer.split("JAI, hello!"), ["JAI", ",", " ", "hello", "!"])

    def test_vocabulary_and_round_trip(self) -> None:
        tokenizer = Tokenizer(); sample = "JAI learns fast."; tokenizer.build_vocabulary([sample])
        ids = tokenizer.encode(sample, add_bos=True, add_eos=True)
        self.assertEqual(ids[0], 2); self.assertEqual(ids[-1], 3)
        self.assertEqual(tokenizer.decode(ids, skip_special=True), sample)

    def test_unknown_token(self) -> None:
        tokenizer = Tokenizer(); tokenizer.build_vocabulary(["hello"])
        self.assertEqual(tokenizer.encode("goodbye"), [1])

    def test_save_and_load(self) -> None:
        tokenizer = Tokenizer(); tokenizer.build_vocabulary(["JAI remembers."])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"; tokenizer.save(path); loaded = Tokenizer.load(path)
            self.assertEqual(loaded.tokens, tokenizer.tokens)
            self.assertEqual(loaded.encode("JAI remembers."), tokenizer.encode("JAI remembers."))

    def test_invalid_token_id(self) -> None:
        with self.assertRaises(ValueError): Tokenizer().decode([999])

if __name__ == "__main__": unittest.main()