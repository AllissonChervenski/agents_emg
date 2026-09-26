import json
import re
from orchestrator.config.models import ValidationIssue
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


def parse_validation(text: str, validator: str, model=None):
    obj = _extract(text)
    if not isinstance(obj, dict) or obj.get("status") not in {"PASS", "REVISE", "BLOCKED"} or not isinstance(obj.get("issues", []), list):
        return validation_result("BLOCKED", [], "Could not parse strict validation output", validator, model, text)
    issues = []
    try:
        for item in obj.get("issues", []):
            issues.append(ValidationIssue(str(item["id"]), str(item["severity"]), str(item.get("artifact", "")), str(item.get("location", "")), str(item.get("requirement", "")), str(item["description"]), str(item.get("suggested_action", ""))))
    except (KeyError, TypeError):
        return validation_result("BLOCKED", [], "Malformed issue schema", validator, model, text)
    return validation_result(obj["status"], issues, str(obj.get("summary", "")), validator, model, text)
