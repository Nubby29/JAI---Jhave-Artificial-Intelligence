# JAI Version: 0.6.0
"""Tests for JAI's chat interface."""

import unittest

from brain.transformer import TransformerLanguageModel
from chat.session import ChatSession
from tokenizer.tokenizer import Tokenizer


class TestChatSession(unittest.TestCase):
    def test_empty_message(self) -> None:
        tokenizer = Tokenizer(["User", ":", "JAI"])
        model = TransformerLanguageModel(len(tokenizer.tokens), model_size=4, context_size=8)
        session = ChatSession(model, tokenizer)
        self.assertIn("Please say something", session.reply("   "))

    def test_chat_history(self) -> None:
        tokenizer = Tokenizer(["User", ":", "JAI"])
        model = TransformerLanguageModel(len(tokenizer.tokens), model_size=4, context_size=8)
        session = ChatSession(model, tokenizer)
        session.history.append(("hello", "hi"))
        self.assertEqual(session.history[0], ("hello", "hi"))


if __name__ == "__main__":
    unittest.main()
