"""Deterministic user pronunciation overrides for project synthesis."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

RuleMode = Literal["literal", "word", "regex"]


@dataclass(frozen=True)
class PronunciationRule:
    match: str
    replacement: str
    mode: RuleMode = "literal"
    case_sensitive: bool = True
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.match:
            raise ValueError("Pronunciation rule match cannot be empty.")
        if self.mode not in {"literal", "word", "regex"}:
            raise ValueError(f"Unsupported pronunciation rule mode: {self.mode}")
        if self.mode == "regex":
            re.compile(self.match)

    def as_dict(self) -> dict[str, Any]:
        return {
            "match": self.match,
            "replacement": self.replacement,
            "mode": self.mode,
            "case_sensitive": self.case_sensitive,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class PronunciationReport:
    text: str
    replacements: int
    applied: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "replacements": self.replacements,
            "applied": list(self.applied),
        }


def _rule_from_mapping(item: Mapping[str, Any]) -> PronunciationRule:
    match = item.get("match", item.get("source", item.get("from", "")))
    replacement = item.get(
        "replacement", item.get("target", item.get("to", item.get("read_as", "")))
    )
    return PronunciationRule(
        match=str(match),
        replacement=str(replacement),
        mode=str(item.get("mode", "literal")).lower(),  # type: ignore[arg-type]
        case_sensitive=bool(item.get("case_sensitive", True)),
        priority=int(item.get("priority", 0)),
    )


def _rules_from_json(value: Any) -> list[PronunciationRule]:
    if isinstance(value, Mapping):
        rules: list[PronunciationRule] = []
        for match, replacement in value.items():
            if isinstance(replacement, Mapping):
                payload = dict(replacement)
                payload.setdefault("match", str(match))
                rules.append(_rule_from_mapping(payload))
            else:
                rules.append(
                    PronunciationRule(match=str(match), replacement=str(replacement))
                )
        return rules
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not all(isinstance(item, Mapping) for item in value):
            raise ValueError("Pronunciation JSON arrays must contain rule objects.")
        return [_rule_from_mapping(item) for item in value]
    raise ValueError("Pronunciation JSON must be an object or an array of rules.")


def parse_pronunciation_rules(
    specification: str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[PronunciationRule, ...]:
    """Parse JSON or simple ``source => spoken form`` lines.

    JSON supports object shorthand and full rule objects. Line syntax accepts a
    tab, ``=>`` or ``=`` delimiter and ignores blank lines and ``#`` comments.
    """

    if specification is None:
        return ()
    if not isinstance(specification, str):
        rules = _rules_from_json(specification)
    else:
        stripped = specification.strip()
        if not stripped:
            return ()
        if stripped.startswith(("{", "[")):
            rules = _rules_from_json(json.loads(stripped))
        else:
            rules = []
            for line_number, raw_line in enumerate(stripped.splitlines(), start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                delimiter = next(
                    (candidate for candidate in ("\t", "=>", "=") if candidate in line),
                    None,
                )
                if delimiter is None:
                    raise ValueError(
                        f"Pronunciation line {line_number} needs a tab, => or = delimiter."
                    )
                match, replacement = line.split(delimiter, 1)
                rules.append(
                    PronunciationRule(
                        match=match.strip(), replacement=replacement.strip()
                    )
                )

    seen: set[tuple[str, str, bool]] = set()
    ordered: list[PronunciationRule] = []
    for rule in sorted(rules, key=lambda item: (-item.priority, -len(item.match))):
        identity = (rule.mode, rule.match, rule.case_sensitive)
        if identity in seen:
            raise ValueError(f"Duplicate pronunciation rule: {rule.match!r}")
        seen.add(identity)
        ordered.append(rule)
    return tuple(ordered)


def apply_pronunciation(
    text: str,
    rules: Sequence[PronunciationRule],
) -> PronunciationReport:
    rendered = str(text)
    applied: list[dict[str, Any]] = []
    total = 0
    for rule in rules:
        pattern = rule.match if rule.mode == "regex" else re.escape(rule.match)
        if rule.mode == "word":
            pattern = rf"(?<!\w){pattern}(?!\w)"
        flags = 0 if rule.case_sensitive else re.IGNORECASE
        rendered, count = re.subn(pattern, rule.replacement, rendered, flags=flags)
        if count:
            total += count
            applied.append({**rule.as_dict(), "count": count})
    return PronunciationReport(rendered, total, tuple(applied))


__all__ = [
    "PronunciationReport",
    "PronunciationRule",
    "apply_pronunciation",
    "parse_pronunciation_rules",
]
