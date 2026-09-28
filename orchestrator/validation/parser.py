import json
import re
from typing import Any
from orchestrator.config.models import ValidationIssue, ValidationResult
from .schemas import validation_result


def _extract(text: str):
    text = text.strip()
    try: return json.loads(text)
    except ValueError: pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try: return json.loads(fenced.group(1))
        except ValueError: pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try: return json.loads(text[start:end+1])
        except ValueError: pass
    return None


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"(?is)<(thought|thinking|reasoning)>.*?</\1>", "", text)
    cleaned = re.sub(r"(?is)^<(thought|thinking|reasoning)>.*", "", cleaned)
    cleaned = re.sub(
        r"(?i)\b(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*['\"]?[^\s,'\"]+['\"]?",
        r"\1=[REDACTED]",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\bbearer\s+[a-zA-Z0-9_\-\.]{8,}", "Bearer [REDACTED]", cleaned)
    cleaned = re.sub(r"\b(ghp_[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9]{20,}|AIza[a-zA-Z0-9_\-]{30,})", "[REDACTED]", cleaned)
    return cleaned.strip()


def _is_codex_jsonl(text: str) -> bool:
    if not text:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    first = lines[0]
    try:
        data = json.loads(first)
        if isinstance(data, dict):
            event_type = data.get("type", "")
            if event_type in {
                "thread.started",
                "turn.started",
                "item.started",
                "item.completed",
                "turn.completed",
            }:
                return True
    except (ValueError, TypeError):
        if first.startswith('{"type":') and any(k in first for k in ('"thread.', '"turn.', '"item.')):
            return True
    return False


def _parse_contract(text: str, validator: str, model: str | None = None) -> ValidationResult:
    obj = _extract(text)
    if not isinstance(obj, dict) or obj.get("status") not in {"PASS", "REVISE", "BLOCKED"} or not isinstance(obj.get("issues", []), list):
        return validation_result("PARSE_ERROR", [], "Could not parse strict validation output", validator, model, text)
    issues = []
    try:
        for item in obj.get("issues", []):
            issues.append(ValidationIssue(str(item["id"]), str(item["severity"]), str(item.get("artifact", "")), str(item.get("location", "")), str(item.get("requirement", "")), str(item["description"]), str(item.get("suggested_action", ""))))
    except (KeyError, TypeError):
        return validation_result("PARSE_ERROR", [], "Malformed issue schema", validator, model, text)
    return validation_result(obj["status"], issues, str(obj.get("summary", "")), validator, model, text)


def parse_codex_validation(text: str, validator: str, model: str | None = None) -> ValidationResult:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return validation_result("PARSE_ERROR", [], "Could not parse strict validation output", validator, model, text)

    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            return validation_result("PARSE_ERROR", [], "Could not parse strict validation output", validator, model, text)
        if not isinstance(event, dict):
            return validation_result("PARSE_ERROR", [], "Could not parse strict validation output", validator, model, text)
        events.append(event)

    agent_messages: list[str] = []
    for event in events:
        if event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                msg_text = item.get("text")
                if isinstance(msg_text, str):
                    agent_messages.append(msg_text)

    if not agent_messages:
        return validation_result("PARSE_ERROR", [], "Could not parse strict validation output", validator, model, text)

    final_text = agent_messages[-1]
    res = _parse_contract(final_text, validator, model)
    res.raw_output = text
    return res


def parse_validation(text: str, validator: str, model: str | None = None, provider: str | None = None) -> ValidationResult:
    if provider == "codex" or (provider is None and _is_codex_jsonl(text)):
        if _is_codex_jsonl(text):
            return parse_codex_validation(text, validator, model)
        if provider == "codex":
            obj = _extract(text)
            if isinstance(obj, dict) and obj.get("status") in {"PASS", "REVISE", "BLOCKED"}:
                return _parse_contract(text, validator, model)
            return parse_codex_validation(text, validator, model)
    return _parse_contract(text, validator, model)
