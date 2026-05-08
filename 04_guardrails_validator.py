"""Step 4 — Guardrails AI Validators (PII redaction + JSON repair)."""

import re
import json

from guardrails import Guard
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
)

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Detect and redact emails, phone numbers, SSNs, and credit card numbers."""

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted = value
        found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in re.findall(pattern, value):
                redacted = redacted.replace(match, f"[{pii_type}_REDACTED]")
                found.append((pii_type, match))

        if found:
            print(f"  !! Redacted {len(found)} PII items: {[p[0] for p in found]}")
            return FailResult(
                error_message=f"PII detected: {[p[0] for p in found]}",
                fix_value=redacted,
            )
        return PassResult()


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Validate and auto-repair malformed JSON."""

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        text = text.replace("'", '"')
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    def validate(self, value: str, metadata: dict):
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            pretty = json.dumps(parsed, indent=2)
            print("  ** JSON repaired successfully")
            return FailResult(
                error_message="Malformed JSON; auto-repaired.",
                fix_value=pretty,
            )
        except json.JSONDecodeError as e:
            return FailResult(
                error_message=f"Invalid JSON after repair attempt: {e}",
                fix_value=json.dumps({"error": str(e), "raw": value}),
            )


def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


def demo_json_guard():
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",      '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",   "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",  '{"key": "value",}'),
        ("Truly invalid",   "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "OK Pass" if result.validation_passed else "FAIL"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:120]}")


def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)
    demo_pii_guard()
    demo_json_guard()
    print("\nOK Step 4 complete!")


if __name__ == "__main__":
    main()
