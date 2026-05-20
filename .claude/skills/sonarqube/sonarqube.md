# SonarQube Quality Gate Checklist

A practical checklist covering the key quality gate conditions to be enforced. Thresholds below are customized per team standards.

---

## 🎯 Custom Quality Gate Thresholds

| Metric                          | Required Threshold        |
| ------------------------------- | ------------------------- |
| **Issues**                      | **= 0**                   |
| **Bugs**                        | **= 0**                   |
| **Security Hotspots Reviewed**  | **≥ 70%**                 |
| **Coverage**                    | **≥ 70%**                 |
| **Duplicated Lines (%)**        | **< 10%**                 |

 ✅ Pull requests / builds must meet **all** the above conditions to pass the quality gate.

---

## Core Metrics (New Code Focus)

Quality gates should primarily evaluate **new code** (code added/modified in the current version), not the entire codebase. This is the "Clean as You Code" approach.

### Reliability
[ ] **Bugs = 0** (no bugs allowed)
[ ] Reliability rating on new code = **A**
[ ] Zero blocker or critical bugs

### Security
[ ] **Issues = 0** (no open issues allowed)
[ ] No new vulnerabilities
[ ] Security rating on new code = **A**
[ ] **Security Hotspots Reviewed ≥ 70%**
[ ] Security review rating acceptable

### Maintainability
[ ] Maintainability rating on new code = **A**
[ ] No new code smells above defined severity

### Coverage
[ ] **Code coverage ≥ 70%**
[ ] Line coverage and branch coverage both tracked
[ ] No uncovered critical paths

### Duplications
[ ] **Duplicated Lines < 10%**
[ ] No large duplicated blocks introduced

---

## Code-Level Checks

[ ] Cyclomatic complexity per function kept low (typically ≤ 15)
[ ] Cognitive complexity flagged and refactored
[ ] File size and function length within reasonable limits
[ ] No commented-out code blocks left behind
[ ] Proper exception handling (no empty catch blocks, no swallowed exceptions)
[ ] No hardcoded credentials, API keys, or secrets
[ ] No TODO / FIXME left unresolved in production merges

---

## Process & Workflow

[ ] Quality gate runs on every pull request, not just main branch
[ ] PR cannot be merged if quality gate fails
[ ] SonarQube integrated with CI/CD (Jenkins, GitHub Actions, GitLab CI, etc.)
[ ] Branch analysis enabled for feature branches
[ ] Results posted as PR comments/decorations for visibility
[ ] Failed gates block deployment pipelines

---

## Language-Specific Rules

[ ] Appropriate rule set activated for each language (Java, JS/TS, Python, C#, etc.)
[ ] Framework-specific rules enabled (Spring, React, Django, etc.)
[ ] Linting aligned between IDE (SonarLint) and SonarQube server

---

## Governance

[ ] Quality profile reviewed and updated quarterly
[ ] Custom rules documented and version-controlled
[ ] Exclusions (generated code, migrations, test fixtures) explicitly defined and justified
[ ] Baseline / leak period configured appropriately (usually "previous version" or "30 days")
[ ] Security hotspots triaged regularly, not just ignored

---

## Common Pitfalls to Avoid

❌ Don't disable rules just to pass the gate; fix the underlying issue or mark as false positive with justification
❌ Don't exclude test code from analysis entirely — tests have quality too
❌ Don't let security hotspots accumulate unreviewed — target ≥ 70% reviewed
❌ Don't ignore bugs or issues — gate requires **zero** of each

---

## Recommended Workflow

1. **Developer writes code** → SonarLint flags issues in IDE in real-time
2. **Pull request opened** → SonarQube analyzes code
3. **Quality gate evaluated** against thresholds:
   - Bugs = 0
   - Issues = 0
   - Security Hotspots Reviewed ≥ 70%
   - Coverage ≥ 70%
   - Duplicated Lines < 10%
4. **If failed** → Developer fixes issues and re-pushes
5. **If passed** → Code review → Merge → Deploy

---

## Summary of Required Gate Conditions

✅ Bugs                         = 0
✅ Issues                       = 0
✅ Security Hotspots Reviewed   ≥ 70%
✅ Coverage                     ≥ 70%
✅ Duplicated Lines (%)         < 10%

---

Last updated: 2026 — Thresholds defined per team quality standards.