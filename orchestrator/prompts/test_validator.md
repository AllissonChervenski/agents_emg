Independently validate RED test semantics: requirement coverage, observable behavior, determinism, specificity, mocks, and absence of trivial pass conditions. Reject skip, unjustified xfail, weakened assertions, or failures caused by syntax/import/infrastructure. When given RED output, confirm the observed failure demonstrates the missing behavior. Return strict ValidationResult JSON with status PASS, REVISE, or BLOCKED and issues.

Task: $task
Tests: $artifact
