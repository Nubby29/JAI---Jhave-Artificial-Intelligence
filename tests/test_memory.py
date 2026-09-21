# JAI Version: 0.9.0
"""Tests for persistent memory and communication-based learning."""
import tempfile
import unittest
from pathlib import Path
from memory.manager import MemoryManager

class MemoryTests(unittest.TestCase):
    def test_remember_creates_raw_and_organized_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            manager=MemoryManager(Path(directory)/"memory"); record=manager.remember("Python can automate tasks.")
            self.assertEqual(record["version"],"0.9.0"); self.assertEqual(manager.stats()["encounters"],1)
            self.assertTrue(list(manager.raw_root.rglob("*.json"))); self.assertTrue(list(manager.organized_root.rglob("*.json")))

    def test_memory_survives_new_manager_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"memory"; MemoryManager(root).remember("JAI should remember that Python is useful.")
            self.assertTrue(MemoryManager(root).search("Python useful"))

    def test_learned_fact_survives_and_is_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"memory"; MemoryManager(root).learn_fact("addition","means","combining numbers")
            facts=MemoryManager(root).learned_facts("addition")
            self.assertEqual(facts[0]["metadata"]["value"],"combining numbers"); self.assertTrue(facts[0]["learned"])

    def test_conversation_memory_can_be_recalled(self):
        with tempfile.TemporaryDirectory() as directory:
            manager=MemoryManager(Path(directory)/"memory"); manager.remember_conversation("My favorite project is JAI.","I will remember this encounter.")
            context=manager.recall_context("favorite project JAI"); self.assertIn("JAI",context); self.assertIn("favorite",context)

if __name__=="__main__": unittest.main()
