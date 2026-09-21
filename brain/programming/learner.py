# JAI Version: 0.14.1
"""General-purpose programming-language knowledge extraction for JAI."""

from __future__ import annotations
import re

class ProgrammingLearner:
    LANGUAGES = {
        "html", "css", "javascript", "typescript", "python", "java", "c",
        "c++", "c#", "php", "ruby", "go", "rust", "swift", "kotlin",
        "sql", "bash", "shell", "json", "xml", "markdown",
    }

    @classmethod
    def detect_language(cls, text: str) -> str | None:
        lowered = text.lower()
        for language in sorted(cls.LANGUAGES, key=len, reverse=True):
            if re.search(rf"(?<![a-z0-9+#]){re.escape(language)}(?![a-z0-9+#])", lowered):
                return language
        return None

    @classmethod
    def extract(cls, text: str) -> list[dict[str, str]]:
        language = cls.detect_language(text)
        if language is None:
            return []
        facts = []

        def add(subject, relation, value, kind="concept"):
            subject = subject.strip(" .!?")
            value = value.strip(" .!?")
            if subject and value:
                facts.append({"language": language, "subject": subject,
                              "relation": relation, "value": value, "kind": kind})

        patterns = [
            (rf"\b{re.escape(language)}\s+stands\s+for\s+(.+?)(?:[.!?]|$)", "stands_for"),
            (rf"\b{re.escape(language)}\s+is\s+used\s+for\s+(.+?)(?:[.!?]|$)", "used_for"),
            (rf"\b{re.escape(language)}\s+is\s+(.+?)(?:[.!?]|$)", "is"),
            (rf"\b{re.escape(language)}\s+means\s+(.+?)(?:[.!?]|$)", "means"),
        ]
        for pattern, relation in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                add(language, relation, match.group(1), "language")

        # HTML declarations such as <!DOCTYPE html> are syntax constructs,
        # not elements, so keep them as a separate programming concept.
        declaration_patterns = [
            rf"(?:in\s+{re.escape(language)}\s*,?\s*)?(?:the\s+)?(<![A-Za-z][^>]*>)\s+(?:declaration\s+)?(?:defines|means|is)\s+(.+?)(?:[.!?]|$)",
        ]
        for pattern in declaration_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                add(f"{language} {match.group(1)}", "defines", match.group(2), "declaration")

        element_patterns = [
            rf"(?:in\s+{re.escape(language)}\s*,?\s*)?(?:the\s+)?(<[a-zA-Z][a-zA-Z0-9-]*>)\s+(?:element\s+)?(?:defines|means|is)\s+(.+?)(?:[.!?]|$)",
            rf"(?:in\s+{re.escape(language)}\s*,?\s*)?(?:the\s+)?(<[a-zA-Z][a-zA-Z0-9-]*>)\s+(?:element\s+)?is\s+used\s+for\s+(.+?)(?:[.!?]|$)",
        ]
        for pattern in element_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                add(f"{language} {match.group(1)}", "defines", match.group(2), "element")

        for snippet in re.findall(r"<![A-Za-z][^>]*>|<[a-zA-Z][^>]*>[^<]*(?:</[a-zA-Z][^>]*>)?", text):
            add(f"{language} syntax", "example", snippet, "syntax")

        return facts
