# JAI Version: 0.12.0
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

    def _is_question(self, text):
        """Prevent questions from being mistaken for facts being taught."""
        normalized = self._normalize(text)
        if not normalized:
            return False
        return (
            text.rstrip().endswith("?")
            or re.match(
                r"^(what|who|where|when|why|how|can|could|do|does|did|is|are|am|will|would|should)\b",
                normalized,
            )
        ) is not None

    def _extract_learning(self, message, force=False, replace=False):
        """Recognize natural teaching statements and store one or more facts."""
        if self.memory is None:
            return None

        text = message.strip()

        if self._is_question(text) and not force:
            return None

        facts = []

        def add(subject, relation, value):
            subject = subject.strip(" .!?")
            value = value.strip(" .!?")
            if subject and value:
                self.memory.learn_fact(subject, relation, value, replace=replace)
                facts.append({"subject": subject, "relation": relation, "value": value})

        # Handle name statements specially so pronouns and relationships are given stable meanings.
        name_match = re.match(r"^(?:your|the)\s+name\s+is\s+(.+?)\.?$", text, re.IGNORECASE)
        possessive_name_match = re.match(
            r"^(my|your|the)\s+(.+?)['’]s\s+name\s+is\s+(.+?)\.?$",
            text,
            re.IGNORECASE,
        )
        if name_match:
            add("JAI", "name", name_match.group(1))
        elif possessive_name_match:
            owner = possessive_name_match.group(1).lower()
            thing = possessive_name_match.group(2).strip()
            value = possessive_name_match.group(3)
            add(f"{owner} {thing}", "name", value)
        else:
            patterns = [
                (r"^(?:please\s+)?remember(?:\s+that)?\s+my\s+name\s+is\s+(.+?)\.?$", "my name", "name"),
                (r"^my\s+name\s+is\s+(.+?)\.?$", "my name", "name"),
                (r"^i\s+am\s+(.+?)\.?$", "user", "identity"),
                (r"^(?:actually|no),?\s+(.+?)\s+is\s+(.+?)\.?$", None, "is"),
                (r"^(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s+means\s+(.+?)\.?$", None, "means"),
                (r"^(.+?)\s+uses\s+(.+?)\.?$", None, "uses"),
                (r"^(.+?)\s+is\s+used\s+for\s+(.+?)\.?$", None, "used_for"),
                (r"^(.+?)\s+is\s+(.+?)\.?$", None, "is"),
                (r"^(.+?)\s+equals\s+(.+?)\.?$", None, "equals"),
            ]

            # Explicit training may contain several simple facts separated by commas
            # or by "and" (for example: "yes means agree, no means disagree").
            chunks = [text]
            if force and ("," in text or re.search(r"\band\b", text, re.IGNORECASE)):
                chunks = [part.strip() for part in re.split(r",|\band\b", text, flags=re.IGNORECASE) if part.strip()]

            for chunk in chunks:
                for pattern, fixed_subject, relation in patterns:
                    match = re.match(pattern, chunk, re.IGNORECASE)
                    if match:
                        if fixed_subject:
                            add(fixed_subject, relation, match.group(1))
                        else:
                            add(match.group(1), relation, match.group(2))
                        break

        equation_matches = re.findall(
            r"(?:(?:example|for\s+example)\s+)?([^=.!?]+?)\s*=\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE,
        )
        for left, right in equation_matches:
            left = left.strip(" .,:;")
            if re.fullmatch(r"[0-9+\-*/()\s]+", left):
                add(left, "equals", right)

        if not facts and force and text:
            self.memory.learn_statement(text, replace=replace)
            facts.append({"subject": "statement", "relation": "learned", "value": text})

        return facts or None

    def _answer_from_learning(self, message):
        """Answer questions using facts explicitly learned through conversation."""
        if self.memory is None:
            return None

        normalized = self._normalize(message)
        # Test a learned "subject is value" fact with a natural yes/no question.
        yes_no = re.match(r"^(?:is|are)\s+(.+?)\s+(.+?)(?:\s+yes\s+or\s+no)?$", normalized)
        if yes_no:
            subject = yes_no.group(1).strip()
            value = yes_no.group(2).strip()
            value = re.sub(r"\s+yes\s+or\s+no$", "", value).strip()
            facts = self.memory.learned_facts(subject)
            for fact in facts:
                meta = fact.get("metadata", {})
                if meta.get("relation") == "is" and self._normalize(str(meta.get("value", ""))) == value:
                    return "Yes."
            if facts:
                return "No."

        patterns = [
            (r"^(?:what(?:'s| is)|who\s+is)\s+my\s+name$", "my name"),
            (r"^(?:what(?:'s| is)|who\s+is)\s+your\s+name$", "JAI"),
            (r"^(?:what|who)\s+is\s+(?:my|your|the)\s+(.+?)\s+s\s+name$", None),
            (r"^(?:what\s+s|whats)\s+(?:my|your|the)\s+(.+?)\s+s\s+name$", None),
            (r"^(?:what\s+is|whats)\s+the\s+name\s+of\s+(my|your|the)\s+(.+?)$", None),
            (r"^(?:who\s+am\s+i)$", "user"),
            (r"^(.+?)\s*=\s*\?$", None),
            (r"^(?:what|who)\s+is\s+(.+?)$", None),
        ]

        subject = None
        for pattern, forced_subject in patterns:
            match = re.match(pattern, normalized)
            if match:
                if forced_subject:
                    subject = forced_subject
                elif pattern.startswith(r"^(?:what|who)\s+is\s+(?:my|your|the)"):
                    query_match = re.match(
                        r"^(?:what|who)\s+is\s+(my|your|the)\s+(.+?)\s+s\s+name$",
                        normalized,
                        re.IGNORECASE,
                    )
                    if query_match:
                        subject = f"{query_match.group(1).lower()} {query_match.group(2).strip()}"
                    else:
                        subject = match.group(1).strip()
                elif pattern.startswith(r"^(?:what\s+s|whats)"):
                    query_match = re.match(
                        r"^(?:what\s+s|whats)\s+(my|your|the)\s+(.+?)\s+s\s+name$",
                        normalized,
                        re.IGNORECASE,
                    )
                    if query_match:
                        subject = f"{query_match.group(1).lower()} {query_match.group(2).strip()}"
                elif pattern.startswith(r"^(?:what\s+is|whats)\s+the\s+name"):
                    query_match = re.match(
                        r"^(?:what\s+is|whats)\s+the\s+name\s+of\s+(my|your|the)\s+(.+?)$",
                        normalized,
                        re.IGNORECASE,
                    )
                    if query_match:
                        subject = f"{query_match.group(1).lower()} {query_match.group(2).strip()}"
                else:
                    subject = match.group(1).strip()
                    # Natural questions often include an article ("a dog", "an animal",
                    # "the sky"), while learned facts are stored by their core subject.
                    subject = re.sub(r"^(?:a|an|the)\s+", "", subject, count=1, flags=re.IGNORECASE)
                break

        if subject is None:
            return None

        facts = self.memory.learned_facts(subject)
        if not facts:
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
        if relation == "uses":
            return f"{learned_subject} uses {value}."
        if relation == "used_for":
            return f"{learned_subject} is used for {value}."
        if relation == "identity":
            return f"You are {value}."
        if relation == "name":
            normalized_subject = self._normalize(learned_subject)
            if normalized_subject == "jai":
                return f"My name is {value}."
            if normalized_subject == "my name":
                return f"Your name is {value}."
            if normalized_subject.startswith("my "):
                thing = learned_subject[3:].strip()
                return f"Your {thing}'s name is {value}."
            if normalized_subject.startswith("your "):
                thing = learned_subject[5:].strip()
                return f"Your {thing}'s name is {value}."
            if normalized_subject.startswith("the "):
                thing = learned_subject[4:].strip()
                return f"The {thing}'s name is {value}."
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

    def _handle_learning_trigger(self, message):
        """Handle explicit --train and --fix commands inside the chat."""
        match = re.match(r"^--(train|fix)\s*(.*)$", message.strip(), re.IGNORECASE | re.DOTALL)
        if not match:
            return None

        mode, payload = match.group(1).lower(), match.group(2).strip()
        if not payload:
            return f"Usage: --{mode} <information to {'learn' if mode == 'train' else 'correct'}>."

        facts = self._extract_learning(payload, force=True, replace=(mode == "fix"))
        if mode == "fix":
            return f"I corrected {len(facts or [])} learned item{'s' if len(facts or []) != 1 else ''}. I will use the updated information."
        return f"I learned {len(facts or [])} item{'s' if len(facts or []) != 1 else ''} from this training instruction."

    def reply(self, message):
        if not message.strip():
            return "Please say something so I have something to respond to."

        triggered = self._handle_learning_trigger(message)
        if triggered is not None:
            self.history.append((message.strip(), triggered))
            if self.memory is not None:
                self.memory.remember_conversation(message.strip(), triggered)
            return triggered

        learned = self._extract_learning(message)
        if learned:
            if len(learned) == 1:
                fact = learned[0]
                if fact["subject"] == "JAI" and fact["relation"] == "name":
                    raw = f"I learned that my name is {fact['value']}. I will remember it."
                elif fact["subject"] == "my name" and fact["relation"] == "name":
                    raw = f"I learned that your name is {fact['value']}. I will remember it."
                elif fact["relation"] == "name" and fact["subject"].lower().startswith(("my ", "your ", "the ")):
                    owner, thing = fact["subject"].split(" ", 1)
                    possessive = "your" if owner.lower() in {"my", "your"} else "the"
                    raw = f"I learned that {possessive} {thing}'s name is {fact['value']}. I will remember it."
                else:
                    raw = f"I learned that {fact['subject']} {fact['relation']} {fact['value']}. I will remember it."
            else:
                raw = f"I understand. You are teaching me {len(learned)} things, and I will remember them."
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
