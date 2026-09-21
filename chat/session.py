# JAI Version: 0.9.1
"""Interactive chat with communication-based learning and persistent memory."""

import re
from chat.corpus import DIALOGUES


class ChatSession:
    """Conversation layer that can learn simple facts and use them later."""

    def __init__(self, model, tokenizer, memory=None, max_new_tokens=32, temperature=0.35, bootstrap=False):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if temperature < 0:
            raise ValueError("temperature cannot be negative.")
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.bootstrap = bootstrap
        self.history = []

    @staticmethod
    def _normalize(text):
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _extract_learning(self, message):
        """Turn common teaching statements into durable structured facts."""
        if self.memory is None:
            return None

        text = message.strip()
        patterns = [
            (r"^(?:please\s+)?remember(?:\s+that)?\s+my\s+name\s+is\s+(.+?)\.?$", "name", "user"),
            (r"^(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s+is\s+(.+?)\.?$", "is", None),
            (r"^(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s+means\s+(.+?)\.?$", "means", None),
            (r"^(?:actually|no),?\s+(.+?)\s+is\s+(.+?)\.?$", "is", None),
            (r"^(?:the\s+)?capital\s+of\s+(.+?)\s+is\s+(.+?)\.?$", "capital_of", None),
            (r"^(.+?)\s+equals\s+(.+?)\.?$", "equals", None),
            (r"^i\s+am\s+(.+?)\.?$", "identity", "user"),
            (r"^my\s+name\s+is\s+(.+?)\.?$", "name", "user"),
        ]

        for pattern, relation, subject_override in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if not match:
                continue

            if subject_override == "user" and relation == "name":
                subject, value = "my name", match.group(1)
            elif subject_override == "user" and relation == "identity":
                subject, value = "user", match.group(1)
            else:
                subject, value = match.group(1), match.group(2)

            subject = subject.strip(" .!?")
            value = value.strip(" .!?")
            if relation == "capital_of":
                subject, value = value, subject

            if len(subject) < 1 or not value:
                continue

            self.memory.learn_fact(subject, relation, value)
            return {"subject": subject, "relation": relation, "value": value}

        return None

    def _answer_from_learning(self, message):
        """Answer questions using facts explicitly learned through conversation."""
        if self.memory is None:
            return None

        normalized = self._normalize(message)
        patterns = [
            (r"^(?:what|who)\s+is\s+(.+?)$", None),
            (r"^(?:what(?:'s| is)\s+my\s+name)$", "my name"),
            (r"^(?:who\s+am\s+i)$", "user"),
        ]

        subject = None
        for pattern, forced_subject in patterns:
            match = re.match(pattern, normalized)
            if match:
                subject = forced_subject or match.group(1).strip()
                break

        if subject is None:
            return None

        facts = self.memory.learned_facts(subject)
        if not facts:
            # Handle punctuation differences such as "2+2" vs "2 + 2".
            target = self._normalize(subject)
            for candidate in self.memory.search(subject, limit=20):
                meta = candidate.get("metadata", {})
                if meta.get("type") != "learned_fact":
                    continue
                if self._normalize(str(meta.get("subject", ""))) == target:
                    facts = [candidate]
                    break

        if not facts:
            return None

        meta = facts[0].get("metadata", {})
        relation = meta.get("relation", "is")
        value = meta.get("value", "")
        learned_subject = meta.get("subject", subject)

        if relation == "capital_of":
            return f"The capital of {learned_subject} is {value}."
        if relation == "means":
            return f"{learned_subject} means {value}."
        if relation == "equals":
            return f"{learned_subject} equals {value}."
        if relation == "identity":
            return f"You are {value}."
        if relation == "name":
            return f"Your name is {value}."
        return f"{learned_subject} is {value}."

    def _bootstrap_reply(self, message):
        query = self._normalize(message)
        if not query:
            return "Please say something so I have something to respond to."

        exact = {self._normalize(user): assistant for user, assistant in DIALOGUES}
        if query in exact:
            return exact[query]

        query_words = set(query.split())
        best_answer = None
        best_score = 0.0
        for user, assistant in DIALOGUES:
            candidate_words = set(self._normalize(user).split())
            if not candidate_words:
                continue
            score = len(query_words & candidate_words) / max(len(query_words | candidate_words), 1)
            if score > best_score:
                best_score = score
                best_answer = assistant

        if best_answer is not None and best_score >= 0.5:
            return best_answer
        return "I am ready to chat, and I am still learning."

    def _build_prompt(self, message):
        sections = []
        if self.memory is not None:
            recalled = self.memory.recall_context(message, limit=5)
            if recalled:
                sections.append(recalled)

        if self.history:
            sections.append(
                "Recent conversation:\n"
                + "\n".join(f"User: {user}\nJAI: {assistant}" for user, assistant in self.history[-3:])
            )

        sections.append(f"User: {message.strip()}\nJAI:")
        return "\n\n".join(sections)

    def reply(self, message):
        if not message.strip():
            return "Please say something so I have something to respond to."

        learned = self._extract_learning(message)
        if learned:
            raw = (
                f"I learned that {learned['subject']} {learned['relation']} "
                f"{learned['value']}. I will remember it."
            )
        else:
            raw = self._answer_from_learning(message)
            if raw is None:
                if self.bootstrap:
                    raw = self._bootstrap_reply(message)
                else:
                    prompt_ids = self.tokenizer.encode(self._build_prompt(message), add_bos=True)
                    generated = self.model.generate(
                        prompt_ids,
                        max_new_tokens=self.max_new_tokens,
                        stop_token_id=self.tokenizer.token_to_id[self.tokenizer.EOS],
                        temperature=self.temperature,
                    )
                    raw = self.tokenizer.decode(generated[len(prompt_ids):], skip_special=True)
                    raw = raw.split("User:", 1)[0].split("JAI:", 1)[0]
                    raw = re.sub(r"\s+", " ", raw).strip() or "I am still learning how to respond to that."

        self.history.append((message.strip(), raw))
        if self.memory is not None:
            self.memory.remember_conversation(message.strip(), raw)
        return raw
