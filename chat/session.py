# JAI Version: 0.9.0
"""Interactive chat with communication-based learning and persistent memory."""

import re
from chat.corpus import DIALOGUES

class ChatSession:
    def __init__(self,model,tokenizer,memory=None,max_new_tokens=32,temperature=0.35,bootstrap=False):
        if max_new_tokens<1: raise ValueError("max_new_tokens must be positive.")
        if temperature<0: raise ValueError("temperature cannot be negative.")
        self.model=model; self.tokenizer=tokenizer; self.memory=memory; self.max_new_tokens=max_new_tokens; self.temperature=temperature; self.bootstrap=bootstrap; self.history=[]

    @staticmethod
    def _normalize(text): return re.sub(r"[^a-z0-9]+"," ",text.lower()).strip()

    def _extract_learning(self,message):
        if self.memory is None: return None
        text=message.strip()
        patterns=[
            (r"^(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s+is\s+(.+?)\.?$","is"),
            (r"^(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s+means\s+(.+?)\.?$","means"),
            (r"^(?:actually|no),?\s+(.+?)\s+is\s+(.+?)\.?$","is"),
            (r"^(?:the\s+)?capital\s+of\s+(.+?)\s+is\s+(.+?)\.?$","capital_of"),
            (r"^(.+?)\s+equals\s+(.+?)\.?$","equals"),
        ]
        for pattern,relation in patterns:
            match=re.match(pattern,text,re.IGNORECASE)
            if not match: continue
            subject=match.group(1).strip(" .!?"); value=match.group(2).strip(" .!?")
            if relation=="capital_of": subject,value=value,subject
            if len(subject)<2 or not value: continue
            self.memory.learn_fact(subject,relation,value)
            return {"subject":subject,"relation":relation,"value":value}
        return None

    def _answer_from_learning(self,message):
        if self.memory is None: return None
        normalized=self._normalize(message)
        match=re.match(r"^(?:what|who) is (.+?)$",normalized)
        if not match: return None
        subject=match.group(1).strip(); facts=self.memory.learned_facts(subject)
        if not facts: return None
        meta=facts[0].get("metadata",{}); relation=meta.get("relation","is"); value=meta.get("value","")
        if relation=="capital_of": return f"The capital of {subject} is {value}."
        if relation=="means": return f"{subject} means {value}."
        if relation=="equals": return f"{subject} equals {value}."
        return f"{subject} is {value}."

    def _bootstrap_reply(self,message):
        query=self._normalize(message)
        if not query: return "Please say something so I have something to respond to."
        exact={self._normalize(user):assistant for user,assistant in DIALOGUES}
        if query in exact: return exact[query]
        q=set(query.split()); best=None; score=0.0
        for user,assistant in DIALOGUES:
            words=set(self._normalize(user).split())
            if words:
                s=len(q&words)/max(len(q|words),1)
                if s>score: score=s; best=assistant
        return best if best is not None and score>=0.5 else "I am ready to chat, but I am still learning."

    def _build_prompt(self,message):
        sections=[]
        if self.memory:
            recalled=self.memory.recall_context(message,limit=5)
            if recalled: sections.append(recalled)
        if self.history:
            sections.append("Recent conversation:\n"+"\n".join(f"User: {u}\nJAI: {a}" for u,a in self.history[-3:]))
        sections.append(f"User: {message.strip()}\nJAI:")
        return "\n\n".join(sections)

    def reply(self,message):
        if not message.strip(): return "Please say something so I have something to respond to."
        learned=self._extract_learning(message)
        if learned:
            raw=f"I learned that {learned['subject']} {learned['relation']} {learned['value']}. I will remember it."
        else:
            raw=self._answer_from_learning(message)
            if raw is None:
                if self.bootstrap: raw=self._bootstrap_reply(message)
                else:
                    prompt_ids=self.tokenizer.encode(self._build_prompt(message),add_bos=True)
                    generated=self.model.generate(prompt_ids,max_new_tokens=self.max_new_tokens,stop_token_id=self.tokenizer.token_to_id[self.tokenizer.EOS],temperature=self.temperature)
                    raw=self.tokenizer.decode(generated[len(prompt_ids):],skip_special=True)
                    raw=raw.split("User:",1)[0].split("JAI:",1)[0]
                    raw=re.sub(r"\s+"," ",raw).strip() or "I am still learning how to respond to that."
        self.history.append((message.strip(),raw))
        if self.memory: self.memory.remember_conversation(message.strip(),raw)
        return raw
