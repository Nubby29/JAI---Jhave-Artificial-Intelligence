# JAI Version: 0.7.0
"""Persistent, organized long-term memory for JAI.

The memory system keeps the original encounter and also creates organized
records that can be recalled later. Raw memories are never overwritten.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryManager:
    """Store, organize, search, and recall JAI's persistent memories."""

    CATEGORIES = (
        "conversations",
        "knowledge",
        "people",
        "concepts",
        "experiences",
        "tasks",
        "skills",
        "procedures",
        "errors",
        "relationships",
    )

    def __init__(self, root: str | Path = "memory") -> None:
        self.root = Path(root)
        self.raw_root = self.root / "raw"
        self.organized_root = self.root / "organized"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.organized_root.mkdir(parents=True, exist_ok=True)
        for category in self.CATEGORIES:
            (self.organized_root / category).mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_json(self.index_path, {
                "version": "0.7.0",
                "encounters": 0,
                "records": 0,
            })

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_name(value: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
        return value.strip("_")[:80] or "memory"

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _read_index(self) -> dict[str, Any]:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"version": "0.7.0", "encounters": 0, "records": 0}

    def _update_index(self, encounter_delta: int = 0, record_delta: int = 0) -> None:
        index = self._read_index()
        index["version"] = "0.7.0"
        index["encounters"] = index.get("encounters", 0) + encounter_delta
        index["records"] = index.get("records", 0) + record_delta
        self._write_json(self.index_path, index)

    def _category_for(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in (
            "error", "failed", "failure", "bug", "broken", "exception", "traceback",
        )):
            return "errors"
        if any(word in lowered for word in (
            "task", "todo", "need to", "have to", "must ", "should ",
        )):
            return "tasks"
        if any(word in lowered for word in (
            "how to", "steps", "procedure", "first ", "then ", "click ", "open ",
        )):
            return "procedures"
        if any(word in lowered for word in (
            "skill", "can do", "learned how", "ability",
        )):
            return "skills"
        if any(word in lowered for word in (
            "friend", "family", "brother", "sister", "mother", "father",
            "relationship", "works with", "belongs to",
        )):
            return "relationships"
        if any(word in lowered for word in (
            "i am", "my name", "my ", "user", "person", "people",
        )):
            return "people"
        if any(word in lowered for word in (
            "i tried", "i did", "worked", "didn't work", "succeeded",
            "success", "failed", "experience",
        )):
            return "experiences"
        if any(word in lowered for word in (
            "what is", "means", "meaning", "concept", "definition",
        )):
            return "concepts"
        return "knowledge"

    def _keywords(self, text: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9']+", text.lower())
        stop_words = {
            "the", "and", "that", "this", "with", "from", "have", "has",
            "what", "when", "where", "which", "would", "could", "should",
            "about", "there", "their", "they", "them", "your", "you",
            "jai", "user", "into", "then", "than", "just", "like",
        }
        seen = set()
        result = []
        for word in words:
            if len(word) < 3 or word in stop_words or word in seen:
                continue
            seen.add(word)
            result.append(word)
        return result[:20]

    def _make_record(
        self,
        category: str,
        encounter_id: str,
        text: str,
        response: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        keywords = self._keywords(text + " " + response)
        summary = text.strip()
        if len(summary) > 500:
            summary = summary[:497] + "..."
        return {
            "version": "0.7.0",
            "memory_id": encounter_id,
            "category": category,
            "created_at": self._timestamp(),
            "content": summary,
            "response": response.strip(),
            "keywords": keywords,
            "confidence": 0.5,
            "times_recalled": 0,
            "metadata": metadata or {},
        }

    def remember(
        self,
        content: str,
        *,
        response: str = "",
        source: str = "encounter",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist an encounter and create an organized memory record."""
        if not content.strip():
            raise ValueError("Memory content cannot be empty.")

        now = datetime.now(timezone.utc)
        encounter_id = now.strftime("%Y%m%dT%H%M%S%fZ")
        category = self._category_for(content + " " + response)
        record = self._make_record(
            category,
            encounter_id,
            content,
            response,
            {
                "source": source,
                **(metadata or {}),
            },
        )

        raw_record = {
            "version": "0.7.0",
            "memory_id": encounter_id,
            "created_at": record["created_at"],
            "source": source,
            "content": content,
            "response": response,
            "organized_category": category,
            "metadata": metadata or {},
        }

        raw_path = self.raw_root / now.strftime("%Y") / now.strftime("%m") / f"{encounter_id}.json"
        organized_name = self._safe_name(record["keywords"][0] if record["keywords"] else category)
        organized_path = (
            self.organized_root
            / category
            / f"{encounter_id}_{organized_name}.json"
        )

        self._write_json(raw_path, raw_record)
        self._write_json(organized_path, record)
        self._update_index(encounter_delta=1, record_delta=1)
        return record

    def remember_conversation(self, user_message: str, jai_response: str) -> dict[str, Any]:
        """Store a complete conversation turn as a permanent encounter."""
        return self.remember(
            user_message,
            response=jai_response,
            source="conversation",
            metadata={"type": "conversation_turn"},
        )

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search organized memory using simple keyword matching."""
        if not query.strip():
            return []
        query_words = set(self._keywords(query))
        if not query_words:
            query_words = set(re.findall(r"[A-Za-z0-9']+", query.lower()))

        scored: list[tuple[int, str, dict[str, Any]]] = []
        for path in self.organized_root.rglob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            haystack = set(record.get("keywords", []))
            haystack.update(self._keywords(record.get("content", "")))
            score = len(query_words & haystack)
            if score:
                scored.append((score, record.get("created_at", ""), record))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        results = [item[2] for item in scored[:max(0, limit)]]

        for record in results:
            self._increment_recall(record.get("memory_id", ""))
        return results

    def _increment_recall(self, memory_id: str) -> None:
        if not memory_id:
            return
        for path in self.organized_root.rglob(f"{memory_id}_*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            record["times_recalled"] = record.get("times_recalled", 0) + 1
            self._write_json(path, record)
            return

    def recall_context(self, query: str, limit: int = 5) -> str:
        """Return recalled memories as compact context for JAI."""
        memories = self.search(query, limit=limit)
        if not memories:
            return ""
        lines = ["Relevant memories:"]
        for memory in memories:
            content = memory.get("content", "").replace("\n", " ")
            response = memory.get("response", "").replace("\n", " ")
            if response:
                lines.append(f"- {content} -> JAI previously replied: {response}")
            else:
                lines.append(f"- {content}")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """Return current persistent memory statistics."""
        index = self._read_index()
        categories = {}
        for category in self.CATEGORIES:
            categories[category] = len(list((self.organized_root / category).glob("*.json")))
        return {
            "version": "0.7.0",
            "root": str(self.root),
            "encounters": index.get("encounters", 0),
            "records": index.get("records", 0),
            "categories": categories,
        }
