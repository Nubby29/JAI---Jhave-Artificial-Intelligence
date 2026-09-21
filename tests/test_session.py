# JAI Version: 0.14.0
"""Tests for communication-based learning, yes/no questions, and learned-answer priority."""

import tempfile
import unittest
from pathlib import Path

from chat.session import ChatSession
from memory.manager import MemoryManager


class DummyModel:
    def generate(self, *args, **kwargs):
        return []


class DummyTokenizer:
    EOS = "<EOS>"

    def encode(self, *args, **kwargs):
        return []

    def decode(self, *args, **kwargs):
        return ""


class ChatSessionLearningTests(unittest.TestCase):
    def test_train_then_natural_question_uses_learned_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryManager(Path(directory) / "memory")
            chat = ChatSession(
                DummyModel(),
                DummyTokenizer(),
                memory=memory,
                bootstrap=True,
            )

            self.assertIn("1 item", chat.reply("--train Dog is an animal"))
            self.assertEqual(chat.reply("What is a dog?"), "Dog is an animal.")

    def test_yes_no_question_uses_learned_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryManager(Path(directory) / "memory")
            chat = ChatSession(DummyModel(), DummyTokenizer(), memory=memory, bootstrap=True)

            chat.reply("--train Elephant is big")
            self.assertEqual(chat.reply("Is elephant big?"), "Yes.")
            self.assertEqual(chat.reply("Is elephant small?"), "No.")
            self.assertEqual(chat.reply("Is elephant big? yes or no"), "Yes.")

    def test_training_can_store_multiple_comma_separated_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryManager(Path(directory) / "memory")
            chat = ChatSession(DummyModel(), DummyTokenizer(), memory=memory, bootstrap=True)

            self.assertIn("2 items", chat.reply("--train yes means agree, no means disagree"))
            self.assertEqual(chat.reply("What is yes?"), "yes means agree.")
            self.assertEqual(chat.reply("What is no?"), "no means disagree.")

    def test_calculation_reasoning_handles_natural_language(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryManager(Path(directory) / "memory")
            chat = ChatSession(DummyModel(), DummyTokenizer(), memory=memory, bootstrap=True)

            self.assertEqual(chat.reply("add 1 and 2"), "3.")
            self.assertEqual(chat.reply("what is 5 plus 7?"), "12.")
            self.assertEqual(chat.reply("subtract 3 from 10"), "7.")
            self.assertEqual(chat.reply("multiply 4 by 5"), "20.")
            self.assertEqual(chat.reply("20 divided by 4"), "5.")

    def test_programming_language_knowledge_and_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryManager(Path(directory) / "memory")
            chat = ChatSession(DummyModel(), DummyTokenizer(), memory=memory, bootstrap=True)

            self.assertIn("1 item", chat.reply("--train HTML is the standard markup language for creating Web pages."))
            self.assertEqual(
                chat.reply("What is HTML?"),
                "HTML is the standard markup language for creating Web pages.",
            )

            chat.reply("--train In HTML, the <p> element defines a paragraph.")
            self.assertEqual(
                chat.reply("What does <p> do?"),
                "<p> defines a paragraph.",
            )

            chat.reply("--train In HTML, <h1> defines the main heading.")
            self.assertIn(
                "h1",
                chat.reply("What does <h1> do?").lower(),
            )

    def test_calculation_reasoning_respects_operator_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryManager(Path(directory) / "memory")
            chat = ChatSession(DummyModel(), DummyTokenizer(), memory=memory, bootstrap=True)

            self.assertEqual(chat.reply("2 + 3 * 4"), "14.")
            self.assertEqual(chat.reply("(2 + 3) * 4"), "20.")

    def test_fix_then_natural_question_uses_updated_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryManager(Path(directory) / "memory")
            chat = ChatSession(
                DummyModel(),
                DummyTokenizer(),
                memory=memory,
                bootstrap=True,
            )

            chat.reply("--train Dog is an animal")
            chat.reply("--fix Dog is a mammal")
            self.assertEqual(chat.reply("What is a dog?"), "Dog is a mammal.")


if __name__ == "__main__":
    unittest.main()
