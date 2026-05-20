# Automatic Execution Protocol

## Overview

This protocol defines how Cursor AI operates **automatically** at every session start without requiring developer prompts. This is the core autonomous behavior of the Temporal Integration System.

---

## Mode of Operation: PROACTIVE and AUTONOMOUS

**No developer prompts needed** → **No confirmation requested** → **Automatic execution**

The system operates in a continuous loop:
- Read state → Validate config → Detect patterns → Implement Temporal → Update state
- Monitor context → Preserve state → Continue seamlessly across sessions

---

## Enhanced Automatic Execution Process (With Codebase Scanning)

When **ANY** chat session starts (including after timeouts), execute these steps automatically:

### STARTUP PHASE (NEW - Steps 1-9)

### Step 1: Dynamic Folder Discovery
**Action:** Discover `cursor-skills-temporal` folder location
**Method:** Search for fixed file names (see `discovery-protocol.md`)
**Result:** Store base path as `{DISCOVERED_PATH}` for all operations

### Step 2: Load System State
**Action:** Read `{DISCOVERED_PATH}/for-cursor/summary.md`
**Purpose:** Load system state and prevent breaking changes
**Critical:** Read this BEFORE any other action to avoid invalid modifications
**Result:** Understand current status, completed components, what NOT to modify

### Step 3: Load Implementation Registry
**Action:** Read `{DISCOVERED_PATH}/progress/temporal-registry.json`
**Purpose:** Load existing implementations to avoid duplicates
**Critical:** Check registry before implementing to verify no similar workflow exists
**Result:** Know what Temporal code already exists in the codebase

### Step 4: Phase 1 Scan - APIs Only (TWO-PHASE WORKFLOW)
**Action:** Run `{DISCOVERED_PATH}/scripts/scan-codebase.py --phase 1` (FAST scan)
**Purpose:** Discover APIs only (internal + external) for quick analysis
**Detects:**
- **Internal APIs:** HTTP routes (REST), GraphQL resolvers, gRPC methods
- **External APIs:** OpenAI, Stripe, AWS, Twilio, etc. (by imports and HTTP calls)
**Performance:** Fast - skips service analysis, only scans for APIs
**Result:** Phase 1 scan report with:
- Internal APIs list (grouped and ranked)
- External APIs list (sorted by call count)
- Summary statistics

### Step 5: Analyze Phase 1 - API Selection (TWO-PHASE WORKFLOW)
**Action:** Run `{DISCOVERED_PATH}/scripts/analyze-temporal.py` with Phase 1 scan report
**Purpose:** Format APIs for developer selection
**Generates:**
- Internal APIs formatted with IDs, types, priorities
- External APIs formatted with call counts, file lists
- Selection prompt
**Result:** Phase 1 analysis report ready for display

### Step 6: Present Phase 1 Results (API SELECTION)
**Action:** Display Phase 1 results in Cursor chat
**Template:** See INTERACTIVE APPROVAL WORKFLOW section in .cursorrules
**Format:**
- Section A: Internal APIs (HTTP routes, GraphQL, gRPC)
- Section B: External APIs (sorted by usage count)
- Selection prompt with examples
**Purpose:** Developer selects which APIs to implement
**Result:** Wait for developer API selection

### Step 7: Wait for API Selection
**Action:** Pause execution until developer responds
**Valid Responses:**
- `APPROVE API 1` - Single API
- `APPROVE API 1,2,5` - Multiple APIs
- `APPROVE ALL INTERNAL` - All internal APIs
- `APPROVE ALL EXTERNAL` - All external APIs
- `DENY` - Skip implementation
**Result:** Selected API IDs or DENY

### Step 8: Phase 2 Scan - Services Using Selected APIs (TARGETED)
**Action:** Run `{DISCOVERED_PATH}/scripts/scan-codebase.py --phase 2 --selected-apis "internal:POST /api/payment,external:OpenAI API"`
**Purpose:** Find all services/functions using selected APIs
**Performance:** Targeted - only scans for usage of selected APIs
**Detects:**
- Services/functions using each API
- Usage frequency per service
- File locations and line numbers
**Result:** Phase 2 scan report with services grouped by API

### Step 9: Analyze Phase 2 - Service Implementation Plan
**Action:** Run `{DISCOVERED_PATH}/scripts/analyze-temporal.py` with Phase 2 scan report
**Purpose:** Generate implementation plan for selected APIs
**Generates:**
- Services grouped by API
- Function usage details
- Implementation strategy prompts
**Result:** Phase 2 analysis report ready for display

### Step 10: Present Phase 2 Results (SERVICE APPROVAL)
**Action:** Display Phase 2 results in Cursor chat
**Template:** See INTERACTIVE APPROVAL WORKFLOW section in .cursorrules
**Shows:**
- Services using each selected API
- Function details with line numbers
- Union vs Intersection prompt (if multiple APIs)
- Shared vs Per-Service prompt (for each API)
**Purpose:** Developer reviews and approves implementation
**Result:** Wait for developer approval and strategy choices

### Step 11: Wait for Strategy & Approval
**Action:** Pause execution until developer responds
**Prompts:**
1. If multiple APIs: Union vs Intersection strategy
2. For each API: Shared vs Per-Service activities
3. Final approval: APPROVE or DENY
**Result:** Implementation strategy and approval/denial

### Step 12: Validate Configuration
**Action:** Automatically validate configuration completeness (see `validation-protocol.md`)
**Purpose:** Ensure all required setup is complete before implementing
**Checks:** Environment variables, dependencies, SDK versions, namespace requirements
**Result:** Either proceed (if valid) or provide complete feedback (if invalid)

### Step 13: Check Registry (Duplicate Prevention)
**Action:** Verify if similar implementation already exists
**Purpose:** Prevent duplicate implementations of same workflow/activity
**Method:** Search registry for similar names, task queues, descriptions
**Result:** Either implement new (if unique) or reference existing (if duplicate)

### Step 14: Implement in temporal/ Folder Structure
**Action:** Generate Temporal code in organized folder structure
**Structure:**
```
temporal/
├── workflows/          # Service workflows
├── activities/         # API activities
│   └── shared/        # Shared activities
├── workers/           # Worker configs
├── client/            # Client setup
├── tests/             # All tests
│   ├── workflows/
│   ├── activities/
│   └── integration/
└── config/            # Configuration
```
**Naming:** Use `[api_name]_activity.py`, `[service_name]_workflow.py`
**Includes:** Retry policies, timeouts, error handling, tests
**Result:** Complete working Temporal implementation

### Step 14A: Generate Unit Tests Automatically (NEW - Phase 1)
**Action:** Run `generate-tests.py` to create unit tests for implementation
**Purpose:** Ensure every Temporal implementation has comprehensive test coverage
**Tests Generated:**
- Workflow tests (happy path, error handling, timeouts)
- Activity tests (success, failure, retry behavior)
- Integration tests (full workflow + activities)
**Test Location:** 
- API (internal): `temporal/tests/api/internal/{api_name}/`
- API (external): `temporal/tests/api/external/{api_name}/`
- Service: `temporal/tests/service/{service_name}/`
**Languages:** Python (pytest), TypeScript (jest), Go (testing), Java (JUnit), .NET (xUnit), PHP (PHPUnit), Ruby (RSpec)
**Result:** Test files created in structured folders

### Step 14B: Run Tests Automatically (NEW - Phase 1)
**Action:** Run `run-tests.py` to execute generated tests
**Purpose:** Verify implementation works correctly before updating registry
**Execution:** Runs tests with appropriate framework for language
**Captures:**
- Pass/fail counts
- Duration
- Coverage percentage
- Per-file breakdown
**Result:** JSON results file with test outcomes

### Step 14C: Create Test Reports (NEW - Phase 1)
**Action:** Run `create-test-reports.py` to generate markdown reports
**Purpose:** Provide human-readable test results and recommendations
**Reports Generated:**
- `REPORT.md` - Individual implementation test report
- `SUMMARY.md` - Overall test status for all implementations
**Contents:**
- Status summary (✅ passed / ❌ failed)
- Test breakdown by file
- Coverage analysis
- Duration metrics
- Recommendations (coverage warnings, failed tests, performance)
**Result:** Markdown reports in test folders

### Step 14D: Verify Tests Passed (NEW - Phase 1)
**Action:** Check if all tests passed
**Purpose:** Prevent broken implementations from being registered
**Logic:**
- If `failed > 0`: Alert developer, DO NOT update registry, return error
- If `failed == 0`: Continue to registry update
**Alert Message (if tests fail):**
```
❌ Tests failed: X/Y tests
Registry will NOT be updated
Fix the failing tests and try again.
```
**Result:** Either proceed (tests passed) or halt (tests failed)
**Critical:** Registry update (Step 20) ONLY happens if this step passes

### MODE DETECTION PHASE (NEW - Phase 2)

### Step 14E: Detect Automation Mode
**Action:** Run `detect-mode.py` to determine which automation scenario applies
**Purpose:** Smart behavior based on codebase state
**Three Modes:**
- **new-module**: No significant code yet → Auto-implement everything
- **post-agent**: Registry with agent flag → Show status, ask for remaining
- **first-time-existing**: Code exists, no registry → Ask once, then auto forever
**Result:** Mode selected with explanation

### Step 14F: Execute Mode-Specific Behavior
**Action:** Follow workflow for detected mode
**Mode 1 (new-module):**
- Auto-implement everything as developer writes
- Show notifications after each implementation
- No approval needed
**Mode 2 (post-agent):**
- Show agent status (implemented vs skipped counts)
- Ask once for remaining LOW priority items
- Auto-implement future code
**Mode 3 (first-time-existing):**
- Run Phase 1+2 with priorities
- Ask once which to implement
- Create registry
- Auto-implement future code
**Result:** Initial setup complete, ready for monitoring

### MONITORING PHASE (Steps 15-19 - Always Active)

### Step 15: Continue Monitoring New Code
**Action:** Monitor developer's code for new patterns as they write
**Purpose:** Maintain proactive detection throughout session
**Result:** Automatic detection when new suitable patterns appear

### Step 16: Detect Patterns (New Code)
**Action:** Automatically detect code patterns that need Temporal implementation
**Purpose:** Identify suitable use cases without waiting for developer request
**Method:** Pattern matching against use cases (see `pattern-detection.md`)
**Result:** Identify which Temporal services fit the developer's code

### Step 17: Create Backup (Deslop Cleanup)
**Action:** Create backup of generated files before cleanup
**Method:** Run `{DISCOVERED_PATH}/scripts/cleanup-code.py --backup-dir .temporal-backups`
**Purpose:** Safety net in case cleanup fails
**Result:** Backup created in `.temporal-backups/` folder

### Step 18: Run Cleanup (Deslop Cleanup)
**Action:** Remove emojis, images, and AI-generated code slop from codebase
**Method:** Run `{DISCOVERED_PATH}/scripts/cleanup-code.py` on repository root
**Scope:** All code files (or diff against main if exists)
**Supports:** All Temporal languages
**Result:** Cleaned code files

### Step 19: Validate Cleanup (Deslop Cleanup)
**Action:** Validate cleaned files are valid
**Checks:** Files exist, readable, no critical errors
**Result:** Validation success or failure

### Step 20: Update Registry (Conditional)
**Action:** Update `{DISCOVERED_PATH}/progress/temporal-registry.json` after implementation
**Condition:** ONLY if cleanup succeeded (Step 19)
**If cleanup failed:** Skip registry update, restore backup, alert developer
**Purpose:** Track all Temporal components created
**Required:** MANDATORY after every workflow, activity, worker, or client implementation (if cleanup succeeds)
**Result:** Registry reflects current state of all Temporal code OR skipped if cleanup failed

### Step 21: Update System State
**Action:** Update `{DISCOVERED_PATH}/for-cursor/summary.md` after modifications
**Purpose:** Maintain state continuity for next session
**Required:** After significant changes or implementations
**Result:** Next session continues seamlessly with full context

---

## Session Start Behavior

### Every Session (Including After Timeout)

**Automatic Actions (No Prompts):**
1. Cursor reads `.cursorrules` automatically (built-in behavior)
2. `.cursorrules` triggers dynamic folder discovery
3. Discovery locates `cursor-skills-temporal` folder
4. System loads state from `summary.md`
5. System loads registry from `temporal-registry.json`
6. System validates configuration
7. System enters monitoring mode

**Developer Experience:**
- No manual setup needed
- No repeated instructions
- System "remembers" everything
- Continues exactly where it left off

---

## State Loading Sequence

**Priority Order:**

1. **Discovery** (Priority 0)
   - Find `cursor-skills-temporal` folder location
   - Store as `{DISCOVERED_PATH}`

2. **System State** (Priority 1)
   - Read `{DISCOVERED_PATH}/for-cursor/summary.md`
   - Understand current status and context

3. **Registry** (Priority 1B)
   - Read `{DISCOVERED_PATH}/progress/temporal-registry.json`
   - Know existing implementations

4. **Validation** (Priority 2)
   - Check configuration completeness
   - Validate dependencies and SDK versions

5. **Knowledge Base** (Priority 3-6)
   - Load relevant `.mdc` files as needed
   - Load reference protocols as needed

---

## Implementation Flow

### Startup Flow (TWO-PHASE WORKFLOW)

**On Session Start:**
1. **Phase 1:** Scan for APIs only (internal + external) - FAST
2. Present APIs to developer → Wait for selection
3. **Phase 2:** Scan for services using selected APIs - TARGETED
4. Present services to developer → Wait for approval & strategy
5. Validate configuration → Implement in temporal/ folder
6. Continue monitoring new code

### When Pattern Detected (New Code)

**If Configuration Valid:**
1. Select appropriate Temporal pattern
2. Generate complete implementation
3. **Create backup** of generated files
4. **Run cleanup** (remove emojis/images)
5. **Validate cleanup** (check files valid)
6. **If cleanup succeeds:** Update registry automatically
7. **If cleanup fails:** Restore backup, skip registry, alert developer
8. Update summary automatically
9. **Silent execution** - no confirmation needed

**If Configuration Invalid:**
1. Generate complete, customized feedback
2. List EVERY missing/invalid item
3. Provide specific commands
4. Wait for developer to fix
5. Resume automatically when fixed

## Interactive Approval Workflow (TWO-PHASE)

**When Scan Completes, Present Analysis with TWO SECTIONS:**

1. **Present Analysis in Chat:**
   ```
   ## 🔍 Temporal Codebase Analysis Complete
   
   ### 📦 SERVICES DETECTED (X found)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1. [Service Name] ([file path])
      - Pattern: [pattern types]
      - Functions: X functions to convert
      - Priority: [HIGH/MEDIUM/LOW]
   
   2. [Service Name] ([file path])
      - Pattern: [pattern types]
      - Functions: X functions to convert
      - Priority: [HIGH/MEDIUM/LOW]
   
   ### 🔌 EXTERNAL APIs DETECTED (X found - sorted by usage)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1. [API Name] (X calls across Y files)
      ├─ [file1]: [function] [line X]
      ├─ [file2]: [function] [line X]
      └─ [file3]: [function] [line X]
      Priority: [HIGH/MEDIUM/LOW]
   
   2. [API Name] (X calls across Y files)
      └─ [usage locations...]
      Priority: [HIGH/MEDIUM/LOW]
   
   ### Efficiency Improvements Available (if Temporal exists):
   💡 [Improvement 1]: [Description]
   💡 [Improvement 2]: [Description]
   
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   
   ### Choose implementation target(s):
   
   Examples:
   APPROVE SERVICE 1           (implement service 1 only)
   APPROVE API 1               (implement all API 1 calls)
   APPROVE SERVICE 1,2 API 3,4 (implement services 1,2 + APIs 3,4)
   APPROVE ALL SERVICES        (implement all services)
   APPROVE ALL APIS            (implement all APIs)
   DENY                        (skip)
   ```

2. **Wait for Developer Response:**
   - Do NOT proceed until developer responds
   - Parse response formats:
     - `APPROVE SERVICE X` - Single service
     - `APPROVE API X` - Single API
     - `APPROVE SERVICE 1,2,3` - Multiple services
     - `APPROVE API 1,2` - Multiple APIs
     - `APPROVE SERVICE 1,2 API 3,4` - Mixed selection
     - `APPROVE ALL SERVICES` - All services
     - `APPROVE ALL APIS` - All APIs
     - `DENY` - Skip all
   - If APPROVE: Proceed with selected target(s)
   - If DENY: Skip and continue monitoring

3. **If API Selected - Ask Implementation Strategy:**
   ```
   You selected: [API Name] (X calls across Y files)
   
   Implementation Strategy:
   A) Shared Activity - One reusable activity for all services
   B) Per-Service Activities - Separate activity per service
   
   Which approach? [A/B]: _
   ```
   - Wait for developer choice (A or B)
   - Proceed with chosen strategy

4. **Implement in temporal/ Folder Structure:**
   - **CRITICAL:** ALL files go in `temporal/` folder:
     ```
     temporal/
     ├── workflows/          # Service workflows
     ├── activities/         # API activities
     │   └── shared/        # Shared activities
     ├── workers/           # Worker configs
     ├── client/            # Client setup
     ├── tests/             # All tests
     │   ├── workflows/
     │   ├── activities/
     │   └── integration/
     └── config/            # Configuration
     ```
   - Use appropriate naming:
     - API activities: `[api_name]_activity.py`
     - Service workflows: `[service_name]_workflow.py`
     - Service activities: `[service]_[function]_activity.py`
   - Generate complete, working code
   - Include retry policies, timeouts, error handling

5. **Implement Only If Approved:**
   - Apply improvements (if selected)
   - Add new services (if SERVICE selected)
   - Implement API activities (if API selected)
   - Refactor existing code (if no Temporal found and approved)
   - Update registry after implementation
   - Update summary.md after changes

---

## State Update Protocol

### After Every Implementation

**Registry Update (MANDATORY):**
- Add workflow entry with full details
- Add activity entries with full details
- Add worker entry with configuration
- Add client entry if created
- Update statistics (counts, languages, task queues)
- Update timestamps

**Summary Update (MANDATORY):**
- Document what was implemented
- Update "Current Implementation State" section
- Update statistics if applicable
- Set `lastUpdated` timestamp
- Note any warnings or context

### Error Handling

**If Registry Update Fails:**
- Log internally and continue
- Don't fail implementation
- Never create new registry file
- Attempt update but proceed if it fails

**If Summary Update Fails:**
- Log internally and continue
- Don't fail implementation
- Never create new summary file
- Attempt update but proceed if it fails

---

## Critical Folder Rules

### Folder Permissions

**READ & UPDATE:**
- `{DISCOVERED_PATH}/for-cursor/` - YOUR knowledge base
  - Read all files
  - Update `summary.md`
  - Read reference files
  
- `{DISCOVERED_PATH}/progress/` - Implementation tracking
  - Read `temporal-registry.json`
  - Update `temporal-registry.json`

**READ ONLY (NEVER MODIFY):**
- `{DISCOVERED_PATH}/for-developer/` - Human documentation
  - Reference only
  - NEVER touch/modify/update/edit
  - Only read for context

**Critical:** Never modify files in `for-developer/` folder - those are for human readers only.

---

## Operating Principles

**PERSISTENCE**
- Instructions apply to EVERY session, FOREVER
- No need for developer to repeat
- Rules are ALWAYS active
- `.cursorrules` = your operating system

**AUTONOMY**
- Implement without asking for permission
- UNLESS configuration is invalid (then provide help)

**VALIDATION**
- Check configuration before EVERY implementation
- Dynamic checking - not hardcoded
- Check EVERYTHING needed for language and namespace
- Provide COMPLETE feedback for ALL missing items

**COMPLETENESS**
- Always generate full, production-ready solutions
- Only when configuration is 100% valid
- Never partial implementations

**CONSISTENCY**
- Read `summary.md` first
- Update after changes
- Maintains state across sessions
- Prevents breaking existing code

**PROACTIVITY**
- Detect and implement before developer asks
- If config valid: implement silently
- If config invalid: alert immediately with complete instructions

---

## Important Notes

**These instructions persist across ALL sessions:**
- You do NOT need developer to repeat them
- Read `.cursorrules` automatically = load these instructions
- Follow automatically every time you see relevant code
- NEVER ask "should I follow these rules?" - they are ALWAYS active

**Developer Experience Goal:**
- Temporal integration happens automatically and invisibly as they code
- No prompts, no questions
- Just working implementations that appear when needed (if configured)
- Or helpful setup instructions (if not configured)

---

**Last Updated:** January 14, 2026 (Added API detection feature: External API scanning, service+API two-level detection, interactive approval with multiple selection, temporal/ folder structure, API implementation strategy selection)

