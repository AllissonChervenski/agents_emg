"""Model family classification and author-validator independence checking."""
from __future__ import annotations

from typing import Iterable


def model_family(model_id: str | None) -> str:
    """Map a model identifier to its fundamental model family.
    
    Model family reflects the underlying model architecture/vendor, NOT the CLI provider.
    For instance, Claude Sonnet and Gemini via AGY belong to different families.
    """
    if not model_id:
        return "unknown"
    low = model_id.lower()
    if any(k in low for k in ("gpt-6", "gpt-5", "sol", "astra", "luna", "o1", "o3")):
        return "gpt"
    if any(k in low for k in ("claude", "sonnet", "opus", "haiku")):
        return "claude"
    if "gemini" in low:
        return "gemini"
    if "mimo" in low:
        return "mimo"
    if "kimi" in low:
        return "kimi"
    if "qwen" in low:
        return "qwen"
    if "deepseek" in low:
        return "deepseek"
    return "other"


def canonical_role(role: str) -> str:
    aliases = {
        "constitution_agent": "constitution",
        "specification_agent": "specification",
        "architect_agent": "planning",
        "task_agent": "tasks",
        "implementation_agent": "coder",
        "cross_artifact_validator": "consistency_agent",
        "clarifier": "clarifier_agent",
    }
    return aliases.get(role, role)


# Mandatory pairs: (author_role, validator_role) where model family MUST NOT collide
MANDATORY_INDEPENDENCE_PAIRS = (
    ("specification", "specification_validator"),
    ("specification", "requirements_reviewer"),
    ("planning", "plan_validator"),
    ("tasks", "tasks_validator"),
    ("test_designer", "test_validator"),
    ("coder", "test_validator"),
    ("coder", "code_reviewer"),
    ("refactorer", "code_reviewer"),
)

# Conditional pairs enforced when numeric_sensitive=True
NUMERIC_INDEPENDENCE_PAIRS = (
    ("test_designer", "code_reviewer"),
    ("test_designer", "coder"),
)


def check_independence(
    author_model: str | None,
    validator_model: str | None,
    pair: tuple[str, str],
    numeric_sensitive: bool = False
) -> tuple[bool, str, str, str]:
    """Check whether author and validator satisfy independence.
    
    Returns (independence_ok, author_family, validator_family, reason).
    """
    auth_fam = model_family(author_model)
    val_fam = model_family(validator_model)
    
    auth_role = canonical_role(pair[0])
    val_role = canonical_role(pair[1])
    canonical_pair = (auth_role, val_role)
    
    is_mandatory = canonical_pair in MANDATORY_INDEPENDENCE_PAIRS
    is_numeric = canonical_pair in NUMERIC_INDEPENDENCE_PAIRS
    
    if not is_mandatory and not is_numeric:
        return True, auth_fam, val_fam, "unconstrained_pair"
        
    if is_numeric and not numeric_sensitive:
        # In normal tasks, log advisory note without blocking
        collision = (auth_fam == val_fam and auth_fam != "unknown")
        reason = "advisory_collision" if collision else "independent_families"
        return True, auth_fam, val_fam, reason

    if auth_fam != "unknown" and auth_fam == val_fam:
        return False, auth_fam, val_fam, f"family_collision: both belong to '{auth_fam}'"
        
    return True, auth_fam, val_fam, "independent_families"


def is_family_independent(
    author_models: str | Iterable[str | None] | None,
    candidate_model: str | None,
    author_role: str | None = None,
    validator_role: str | None = None,
    numeric_sensitive: bool = False,
) -> tuple[bool, str]:
    """Verify that candidate_model does not collide in family with any author_models."""
    if not candidate_model:
        return True, "no_candidate_model"
    if not author_models:
        return True, "no_author_models"
        
    if isinstance(author_models, str):
        auth_list = [author_models]
    else:
        auth_list = [m for m in author_models if m is not None]
        
    cand_fam = model_family(candidate_model)
    if cand_fam in ("unknown", "other"):
        return True, "unclassified_candidate"
        
    pair = (author_role or "", validator_role or "")
    for auth_m in auth_list:
        auth_fam = model_family(auth_m)
        if auth_fam in ("unknown", "other"):
            continue
        ok, _, _, reason = check_independence(
            auth_m, candidate_model, pair, numeric_sensitive=numeric_sensitive
        )
        if not ok:
            return False, reason
            
    return True, "independent"
