# JAI Version: 0.13.0
"""Safe arithmetic reasoning for JAI.

This module deliberately avoids eval/exec. It tokenizes a small arithmetic
language and evaluates it with normal operator precedence.
"""

from __future__ import annotations

import math
import re


class CalculationError(ValueError):
    """Raised when JAI cannot safely parse or calculate an expression."""


class Calculator:
    """Parse and calculate +, -, *, / and parentheses."""

    _TOKEN_RE = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([+\-*/()]))")

    @classmethod
    def calculate(cls, expression: str) -> int | float:
        tokens = cls._tokenize(expression)
        if not tokens:
            raise CalculationError("empty expression")

        values: list[float] = []
        operators: list[str] = []

        def precedence(op: str) -> int:
            return 2 if op in {"*", "/"} else 1

        def apply_operator() -> None:
            if not operators or len(values) < 2:
                raise CalculationError("invalid expression")
            op = operators.pop()
            right = values.pop()
            left = values.pop()
            if op == "+":
                values.append(left + right)
            elif op == "-":
                values.append(left - right)
            elif op == "*":
                values.append(left * right)
            elif op == "/":
                if right == 0:
                    raise CalculationError("division by zero")
                values.append(left / right)

        previous = "operator"
        for token in tokens:
            if isinstance(token, float):
                if previous == "number":
                    raise CalculationError("missing operator")
                values.append(token)
                previous = "number"
            elif token == "(":
                if previous == "number":
                    raise CalculationError("missing operator")
                operators.append(token)
                previous = "operator"
            elif token == ")":
                if previous != "number":
                    raise CalculationError("invalid parentheses")
                while operators and operators[-1] != "(":
                    apply_operator()
                if not operators:
                    raise CalculationError("unmatched parenthesis")
                operators.pop()
                previous = "number"
            else:
                if token == "-" and previous == "operator":
                    # Unary minus is represented as 0 - value.
                    values.append(0.0)
                elif previous != "number":
                    raise CalculationError("invalid operator placement")
                while (
                    operators
                    and operators[-1] != "("
                    and precedence(operators[-1]) >= precedence(token)
                ):
                    apply_operator()
                operators.append(token)
                previous = "operator"

        if previous != "number":
            raise CalculationError("expression cannot end with an operator")

        while operators:
            if operators[-1] == "(":
                raise CalculationError("unmatched parenthesis")
            apply_operator()

        if len(values) != 1 or not math.isfinite(values[0]):
            raise CalculationError("invalid result")

        result = values[0]
        if result.is_integer():
            return int(result)
        return result

    @classmethod
    def _tokenize(cls, expression: str) -> list[float | str]:
        text = expression.strip()
        position = 0
        tokens: list[float | str] = []

        while position < len(text):
            match = cls._TOKEN_RE.match(text, position)
            if not match:
                raise CalculationError("unsupported characters")
            number, operator = match.groups()
            tokens.append(float(number) if number is not None else operator)
            position = match.end()

        return tokens

    @classmethod
    def from_language(cls, message: str) -> tuple[str, int | float] | None:
        """Recognize common natural-language arithmetic requests."""
        text = message.strip().lower().rstrip("?.!")
        patterns = [
            (r"^(?:add|sum)\s+(.+?)\s+and\s+(.+)$", lambda a, b: f"{a}+{b}"),
            (r"^(.+?)\s+plus\s+(.+)$", lambda a, b: f"{a}+{b}"),
            (r"^(.+?)\s+minus\s+(.+)$", lambda a, b: f"{a}-{b}"),
            (r"^(.+?)\s+times\s+(.+)$", lambda a, b: f"{a}*{b}"),
            (r"^(.+?)\s+multiplied\s+by\s+(.+)$", lambda a, b: f"{a}*{b}"),
            (r"^(.+?)\s+divided\s+by\s+(.+)$", lambda a, b: f"{a}/{b}"),
            (r"^divide\s+(.+?)\s+by\s+(.+)$", lambda a, b: f"{a}/{b}"),
            (r"^subtract\s+(.+?)\s+from\s+(.+)$", lambda a, b: f"{b}-{a}"),
            (r"^(?:what is|calculate|compute)\s+(.+)$", lambda a: a),
        ]

        for pattern, builder in patterns:
            match = re.match(pattern, text)
            if not match:
                continue
            try:
                expression = builder(*match.groups())
                if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
                    return None
                return expression, cls.calculate(expression)
            except CalculationError:
                return None

        if re.fullmatch(r"[0-9+\-*/().\s]+", text):
            try:
                return text, cls.calculate(text)
            except CalculationError:
                return None

        return None
