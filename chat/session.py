# JAI Version: 0.6.0
"""Interactive chat session for JAI."""

import re


class ChatSession:
    """Wrap a trained Transformer in a simple user/JAI conversation."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 32, temperature: float = 0.35):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if temperature < 0:
            raise ValueError("temperature cannot be negative.")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.history: list[tuple[str, str]] = []

    def reply(self, message: str) -> str:
        if not message.strip():
            return "Please say something so I have something to respond to."

        prompt = "User: " + message.strip() + "\nJAI:"
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        generated = self.model.generate(
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            stop_token_id=self.tokenizer.token_to_id[self.tokenizer.EOS],
            temperature=self.temperature,
        )
        new_ids = generated[len(prompt_ids):]
        raw = self.tokenizer.decode(new_ids, skip_special=True)
        raw = raw.split("User:", 1)[0].split("JAI:", 1)[0]
        raw = re.sub(r"\s+", " ", raw).strip()
        if not raw:
            raw = "I am still learning how to respond to that."
        self.history.append((message.strip(), raw))
        return raw
