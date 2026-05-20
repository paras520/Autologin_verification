# Code Flow - How Files Work Together (v2.1)

**Last Updated:** February 25, 2026  
**System:** Master Orchestrator + Temporal Skill Architecture with Runtime Control  
**Version:** 2.1

---

## 🎯 Purpose of This Document

This document explains:
- How the new Master Orchestrator works
- How skills are executed
- Data flow through the system
- Which file calls which file
- Where data is stored and retrieved

---

## 📊 System Architecture (v2.0)

```
┌─────────────────────────────────────────────────────────────┐
│           MASTER ORCHESTRATOR (root/.cursorrules)            │
│  - AGENT_MODE control                                        │
│  - Context detection (AGENT vs DEV)                          │
│  - Skill execution routing                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
            ┌─────────────────┴─────────────────┐
            │                                   │
       AGENT Context                       DEV Context
    (Cloud Agent)                    (Developer IDE)
            │                                   │
            ↓                                   ↓
    Check AGENT_MODE                    Show interactive
    If false: Exit                      prompts and menus
    If true: Execute skills             Wait for approval
            ↓                                   ↓
            └─────────────────┬─────────────────┘
                              ↓
                    Execute Enabled Skills
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              TEMPORAL SKILL (skills/temporal/)               │
│  - Skill-specific .cursorrules                               │
│  - summary.md (skill state)                                  │
│  - temporal-registry.json (implementations)                  │
│  - scripts/*.py (automation)                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Two-Phase Workflow:
                              ↓
┌─────────────────────────────────────────────────────────────┐
│         PHASE 1: API Scanning (scan-codebase.py)             │
│  - Detect internal APIs                                      │
│  - Detect external APIs                                      │
│  - Assign priority scores                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Wait for API selection
                              ↓
┌─────────────────────────────────────────────────────────────┐
│    PHASE 2: Service Analysis (deep-scanner.py)               │
│  - Find services using selected APIs                         │
│  - Trace call chains (5 levels deep)                         │
│  - Map dependencies                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Wait for approval
                              ↓
┌─────────────────────────────────────────────────────────────┐
│         IMPLEMENTATION (refactor-to-temporal.py)             │
│  - Generate workflows, activities, workers                   │
│  - Generate unit tests (generate-tests.py)                   │
│  - Run tests (run-tests.py)                                  │
│  - Create reports (create-test-reports.py)                   │
│  - Update registry (temporal-registry.json)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    (If AGENT Context)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              POST-COMPLETION (Agent Only)                    │
│  - 8-step validation                                         │
│  - Set AGENT_MODE = false                                    │
│  - Mark registry: processed_by_agent = true                  │
│  - Create PR (if validation passes)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Detailed Flow: Session Start to Implementation

### **Step 1: Cursor Session Starts (DEV) or Agent Triggers (AGENT)**

```
Developer opens project in Cursor IDE  OR  Cloud agent is triggered
    ↓
Cursor reads: .cursorrules (at root - Master Orchestrator)
    ↓
Master orchestrator detects context:
  - invoked_by == "cursor-cloud-agent" → AGENT Context
  - invoked_by == "cursor-ide" → DEV Context
```

**Files Involved:**
- `.cursorrules` (root - Master Orchestrator)

---

### **Step 2A: AGENT Context (Cloud Agent)**

```
Context = AGENT
    ↓
Master orchestrator checks: AGENT_MODE
    ↓
If AGENT_MODE == false:
  ↓
  Print: "⚠️ AGENT_MODE is false. Agent execution disabled."
  Exit (no further action)
    ↓
If AGENT_MODE == true:
  ↓
  Proceed with autonomous execution
  For each enabled skill:
    ↓
    execute_skill(skill_name)
```

**Files Involved:**
- `.cursorrules` (root - reads AGENT_MODE)

---

### **Step 2B: DEV Context (Developer IDE)**

```
Context = DEV
    ↓
Check registry: skills/temporal/progress/temporal-registry.json
    ↓
If registry has "processed_by_agent": true:
  ↓
  Show agent report:
  "📊 Agent Processing Report
   ✅ Implemented: 20 HIGH priority items
   ⏭️ Skipped: 10 LOW priority items
   
   Would you like to implement skipped items? [Y/N]"
  ↓
  Wait for response
    ↓
Else (no agent processing):
  ↓
  Show welcome message:
  "👋 Welcome to Cursor Skills!
   Available skills:
   - Temporal Integration
   
   Would you like to scan your codebase? [Y/N]"
  ↓
  Wait for response
```

**Files Involved:**
- `.cursorrules` (root)
- `skills/temporal/progress/temporal-registry.json` (reads for agent flag)

---

### **Step 3: Execute Skill (Temporal)**

```
master_orchestrator.execute_skill("temporal")
    ↓
Navigate to: skills/temporal/
    ↓
Read: skills/temporal/.cursorrules (558-line skill-specific rules)
    ↓
Load skill state:
  - skills/temporal/summary.md
  - skills/temporal/progress/temporal-registry.json
    ↓
Execute skill workflow (see Step 4)
```

**Files Involved:**
- `skills/temporal/.cursorrules` (skill-specific rules)
- `skills/temporal/summary.md` (skill state)
- `skills/temporal/progress/temporal-registry.json` (existing implementations)

---

### **Step 4: Detect Automation Mode (Temporal Skill)**

```
Skill executes: python skills/temporal/scripts/detect-mode.py
    ↓
Script checks:
  - Does code exist? (scan src/, app/, etc.)
  - Does registry exist? (check temporal-registry.json)
  - Is processed_by_agent: true?
    ↓
Returns mode:
  - "new-module" (no code yet)
  - "post-agent" (registry with agent flag)
  - "first-time-existing" (code exists, no registry)
    ↓
System adapts behavior based on mode
```

**Files Involved:**
- `skills/temporal/scripts/detect-mode.py` (logic)
- `skills/temporal/progress/temporal-registry.json` (input)
- Developer's codebase (input)

---

### **Step 5: Two-Phase Workflow**

#### **Phase 1: API Scanning**

```
Skill executes: python skills/temporal/scripts/scan-codebase.py --phase 1
    ↓
Script scans:
  - Internal APIs (HTTP routes, GraphQL, gRPC)
  - External APIs (OpenAI, Stripe, AWS, etc.)
  - Imports, config files, HTTP calls
    ↓
Script writes: /tmp/scan-results-phase1.json
    ↓
Skill executes: python skills/temporal/scripts/priority-analyzer.py
    ↓
Reads: /tmp/scan-results-phase1.json
Analyzes complexity, assigns priority scores (HIGH/LOW)
Writes: /tmp/scan-results-phase1-prioritized.json
    ↓
If DEV Context:
  ↓
  Present to developer:
  "HIGH Priority APIs: ..."
  "LOW Priority APIs: ..."
  Wait for selection: "APPROVE API 1,2,5"
    ↓
If AGENT Context:
  ↓
  Auto-select: All HIGH priority APIs
  No wait, proceed immediately
```

**Files Involved:**
- `skills/temporal/scripts/scan-codebase.py` (scanner)
- `skills/temporal/scripts/priority-analyzer.py` (scoring)
- `temporal_scan_results.json` (output - saved to disk)
- `temporal_priorities.json` (output - via priority-analyzer.py --output flag)

#### **Phase 2: Service Scanning**

```
Developer selects APIs: "APPROVE API 1,2,5"  OR  Agent auto-selects HIGH
    ↓
System parses: selected_apis = [1, 2, 5]
    ↓
Skill executes: python skills/temporal/scripts/scan-codebase.py --phase 2 --selected-apis 1,2,5
    ↓
Script scans:
  - All services/functions using selected APIs
  - Usage patterns, call chains (5 levels deep)
  - Deep analysis (services, repositories, async methods)
    ↓
Script writes: /tmp/scan-results-phase2.json
    ↓
If DEV Context:
  ↓
  Present:
  "Services using OpenAI API: 2 services, 20 calls"
  "Services using Stripe API: 1 service, 8 calls"
  Ask strategy: "Union vs Intersection?"
  Ask activity type: "Shared vs Per-Service?"
  Wait for approval: "APPROVE"
    ↓
If AGENT Context:
  ↓
  Auto-select: Union strategy, Shared activities
  No wait, proceed immediately
```

**Files Involved:**
- `skills/temporal/scripts/scan-codebase.py` (scanner)
- `skills/temporal/scripts/deep-scanner.py` (deep analysis)
- `temporal_scan_results.json` (updated with Phase 2 data)

---

### **Step 6: Configuration Validation**

```
Skill executes: python skills/temporal/scripts/validate-system.py
    ↓
Script checks:
  - .env file exists
  - Required variables: TEMPORAL_URI, TEMPORAL_NAMESPACE, etc.
  - Dependencies installed (package.json, requirements.txt, etc.)
  - SDK version (fetches from docs.temporal.io)
    ↓
If ANYTHING missing:
  ↓
  Returns: detailed_feedback.json
  {
    "valid": false,
    "missing_vars": ["TEMPORAL_API_KEY"],
    "missing_deps": ["@temporalio/client"],
    "outdated_sdks": ["Python SDK 1.4.0 → Update to 1.6.0"]
  }
  ↓
  System shows:
  "⚠️ Configuration incomplete: [details]"
  STOP (do not implement)
    ↓
If ALL valid:
  ↓
  Proceed to implementation
```

**Files Involved:**
- `skills/temporal/scripts/validate-system.py` (validator)
- `.env` (input)
- `package.json` / `requirements.txt` / etc. (input)

---

### **Step 7: Implementation**

```
All checks passed, developer approved (or agent auto-approved)
    ↓
Skill executes: python skills/temporal/scripts/refactor-to-temporal.py \
                   api "API_NAME" \
                   temporal_scan_results.json \
                   . \
                   --strategy shared \
                   --agent-mode
    ↓
Script reads:
  - temporal_scan_results.json (what to implement)
  - skills/temporal/temporal-workflows.mdc (patterns)
  - skills/temporal/temporal-activities.mdc (patterns)
  - skills/temporal/temporal-language-examples.mdc (code examples)
  - .env (configuration, including TEMPORAL_STATE)
    ↓
Script generates:
  temporal/
  ├── workflows/
  │   ├── payment_service_workflow.py
  │   └── llm_service_workflow.py
  ├── activities/
  │   └── shared/
  │       ├── openai_activity.py
  │       └── stripe_activity.py
  ├── workers/
  │   └── main_worker.py
  ├── client/
  │   └── temporal_client.py
  └── config/
      └── temporal_config.py

IMPORTANT (v2.1): Runtime Kill Switch Implementation
    ↓
Script ALSO generates wrapper code that checks TEMPORAL_STATE:
    ↓
Example wrapper (PaymentController.java):
  - Checks env var: TEMPORAL_STATE=ON|OFF
  - If OFF → Execute service.processPaymentDirect(request)
  - If ON → Execute temporalClient.executePaymentWorkflow(request)
    ↓
This allows instant enable/disable WITHOUT code redeployment
```

**Autonomous Agent Usage:**
- Use `--agent-mode` flag to skip interactive prompts
- Use `--strategy <shared|per_service>` to specify implementation strategy
- Script reads from `temporal_scan_results.json` (created by scan-codebase.py)
- All prompts are bypassed when flags are provided

**Runtime Kill Switch (NEW in v2.1):**
- ALL generated implementations include environment variable bypass logic
- Controllers/Services check `TEMPORAL_STATE` before executing
- `ON` → Execute via Temporal workflow
- `OFF` → Bypass Temporal, execute original method directly
- NO redeployment required to toggle - just edit .env and restart

**Files Involved:**
- `skills/temporal/scripts/refactor-to-temporal.py` (implementation engine - now supports --agent-mode and --strategy flags)
- `skills/temporal/temporal-*.mdc` (patterns - including kill switch examples)
- `.env` (config - including TEMPORAL_STATE)
- `temporal_scan_results.json` (input - from scan-codebase.py)

---

### **Step 8: Unit Test Generation**

```
Implementation complete
    ↓
Skill executes: python skills/temporal/scripts/generate-tests.py \
                   --manifest /tmp/implementation-manifest.json
    ↓
Script reads:
  - /tmp/implementation-manifest.json (what was implemented)
  - skills/temporal/temporal-testing.mdc (test patterns)
  - temporal/workflows/*.py (actual code to test)
  - temporal/activities/*.py (actual code to test)
    ↓
Script generates:
  temporal/tests/
  ├── workflows/
  │   ├── test_payment_service_workflow.py
  │   └── test_llm_service_workflow.py
  ├── activities/
  │   ├── test_openai_activity.py
  │   └── test_stripe_activity.py
  ├── workers/
  │   └── test_main_worker.py
  └── integration/
      └── test_end_to_end.py
    ↓
Script writes: /tmp/test-manifest.json
```

**Files Involved:**
- `skills/temporal/scripts/generate-tests.py` (test generator)
- `skills/temporal/temporal-testing.mdc` (test patterns)
- `/tmp/implementation-manifest.json` (input)
- `/tmp/test-manifest.json` (output)

---

### **Step 9: Test Execution**

```
Tests generated
    ↓
Skill executes: python skills/temporal/scripts/run-tests.py \
                   --test-manifest /tmp/test-manifest.json
    ↓
Script executes:
  pytest temporal/tests/ -v --json-report --json-report-file=/tmp/test-results.json
    ↓
Captures:
  - Total tests run
  - Passed tests
  - Failed tests
  - Error messages
  - Coverage %
    ↓
Script writes: /tmp/test-results.json
```

**Files Involved:**
- `skills/temporal/scripts/run-tests.py` (test executor)
- `/tmp/test-manifest.json` (input)
- `/tmp/test-results.json` (output)

---

### **Step 10: Test Reporting**

```
Tests executed
    ↓
Skill executes: python skills/temporal/scripts/create-test-reports.py \
                   --results /tmp/test-results.json \
                   --manifest /tmp/implementation-manifest.json
    ↓
Script generates:
  temporal/tests/REPORT.md
  temporal/tests/SUMMARY.md
```

**Files Involved:**
- `skills/temporal/scripts/create-test-reports.py` (report generator)
- `/tmp/test-results.json` (input)
- `temporal/tests/REPORT.md` (output)
- `temporal/tests/SUMMARY.md` (output)

---

### **Step 11: Registry Update**

```
All tests passed
    ↓
Skill executes: python skills/temporal/scripts/refactor-to-temporal.py --update-registry
    ↓
Script updates: skills/temporal/progress/temporal-registry.json
    ↓
Adds:
  {
    "api": "POST /api/payment",
    "workflow": "temporal/workflows/payment_service_workflow.py",
    "priority": "HIGH",
    "priority_score": 85,
    "tests": {
      "total": 15,
      "passed": 15,
      "coverage": 92.5
    }
  }
```

**Files Involved:**
- `skills/temporal/scripts/refactor-to-temporal.py` (registry updater)
- `skills/temporal/progress/temporal-registry.json` (output - CRITICAL!)

---

### **Step 12: Post-Completion (AGENT Context Only)**

```
If context == AGENT:
  ↓
  Run 8-step validation (see below)
  ↓
  If all validations pass:
    ↓
    Set AGENT_MODE = false (in root/.cursorrules)
    Mark registry: "processed_by_agent": true
    Commit changes
    Create PR (assign to last committer)
  ↓
  If any validation fails:
    ↓
    Alert developer
    Do NOT create PR
    Keep AGENT_MODE = true (allow retry)
```

**Files Involved:**
- `.cursorrules` (root - updates AGENT_MODE)
- `skills/temporal/progress/temporal-registry.json` (updates processed_by_agent flag)

---

## 🔍 8-Step Validation (AGENT Context)

```bash
1. Test Pass Rate = 100%? (check /tmp/test-results.json)
2. Build Successful? (mvn clean install / npm run build)
3. JAR Integrity? (Java: size > 1KB, entries > 5)
4. Code Quality? (linting passed)
5. Temporal Files Exist? (workflows, activities, tests)
6. Registry Updated? (implementations count > 0)
7. Test Reports Generated? (REPORT.md, SUMMARY.md exist)
8. No Compilation Errors? (build check)

If ALL pass: Create PR
If ANY fail: Exit without PR
```

---

## 📝 File Dependency Map (v2.0)

```
.cursorrules (root - Master Orchestrator)
  ├─→ Check AGENT_MODE (reads)
  ├─→ Detect context (AGENT vs DEV)
  ├─→ skills/temporal/.cursorrules (executes)
  └─→ Post-completion tasks (if AGENT)

skills/temporal/.cursorrules (Skill-Specific Rules)
  ├─→ skills/temporal/summary.md (reads)
  ├─→ skills/temporal/progress/temporal-registry.json (reads)
  ├─→ skills/temporal/scripts/detect-mode.py (executes)
  ├─→ skills/temporal/scripts/scan-codebase.py (executes)
  ├─→ skills/temporal/scripts/priority-analyzer.py (executes)
  ├─→ skills/temporal/scripts/validate-system.py (executes)
  ├─→ skills/temporal/scripts/refactor-to-temporal.py (executes)
  ├─→ skills/temporal/scripts/generate-tests.py (executes)
  ├─→ skills/temporal/scripts/run-tests.py (executes)
  └─→ skills/temporal/scripts/create-test-reports.py (executes)

skills/temporal/scripts/refactor-to-temporal.py
  ├─→ skills/temporal/temporal-*.mdc (reads patterns)
  ├─→ .env (reads config)
  ├─→ /tmp/scan-results-phase2.json (reads)
  ├─→ temporal/* (writes implementations)
  ├─→ /tmp/implementation-manifest.json (writes)
  └─→ skills/temporal/progress/temporal-registry.json (updates)
```

---

## 🔐 Data Files (Critical!)

### **Persistent Data (Version Controlled):**
- `.cursorrules` (root - Master Orchestrator) - MUST be committed
- `skills/temporal/.cursorrules` (Skill rules) - MUST be committed
- `skills/temporal/progress/temporal-registry.json` - MUST be committed
- `skills/temporal/summary.md` - MUST be committed

### **Temporary Data (Not Version Controlled):**
- `/tmp/scan-results-phase1.json`
- `/tmp/scan-results-phase2.json`
- `/tmp/implementation-manifest.json`
- `/tmp/test-manifest.json`
- `/tmp/test-results.json`

### **Output Data (Version Controlled):**
- `temporal/**/*` (all generated code)
- `temporal/tests/REPORT.md`
- `temporal/tests/SUMMARY.md`

---

## 🎯 Key Takeaways (v2.0)

1. **Master Orchestrator is the brain** - Controls AGENT_MODE and routing
2. **Skills are modular** - Each skill has its own `.cursorrules` and state
3. **AGENT_MODE is the cost controller** - Prevents unwanted AI execution
4. **Context detection is automatic** - AGENT vs DEV behavior
5. **Validation is comprehensive** - 8 steps before PR creation
6. **Registry is the ledger** - Tracks all implementations per skill
7. **Tests are mandatory** - No registry update without passing tests

---

**This flow ensures:**
- ✅ Cost optimization (AGENT_MODE toggle)
- ✅ No duplicate implementations (registry check)
- ✅ No configuration errors (validation)
- ✅ No untested code (mandatory tests)
- ✅ Persistent knowledge (summary + registry per skill)
- ✅ Automatic monitoring (continuous detection)
- ✅ Safe autonomous execution (8-step validation)

---

**System Version:** 2.0 (Master Orchestrator + Skills)  
**Status:** Production Ready ✅
