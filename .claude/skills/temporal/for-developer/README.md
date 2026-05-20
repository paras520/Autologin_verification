# Temporal Integration Knowledge Base

Automatic Temporal workflow implementation via Cursor AI.

---

## For Human Readers: What Is This System?

This repository enables Cursor AI to automatically implement Temporal workflows in your code. It's an intelligent documentation system that teaches Cursor AI to write Temporal code for you.

**What It Does:**
- **Works in NEW and EXISTING codebases** - Scans entire codebase on startup
- **Smart automation with 3 modes** - Auto-implements based on codebase state
- **Mandatory unit testing** - Generates, runs, and reports tests automatically
- **Org-wide processing** - Overnight batch processing across 100+ repos
- Monitors your code as you write it
- Detects patterns that would benefit from Temporal
- Detects existing Temporal implementations and analyzes efficiency
- Automatically implements complete workflow solutions
- Automatically cleans code (removes emojis/images) after implementation
- Includes error handling, testing, and deployment code
- Uses pre-filled Temporal credentials (shared dev server)
- Uses your environment configuration automatically

**What You Get:**
- Automatic background job processing
- Built-in retry logic and error handling
- Distributed transaction support (saga pattern)
- Microservice orchestration
- Scheduled task execution
- Long-running workflow management
- State persistence across failures

**How It Works:**
1. You run quick-start script (creates .env with pre-filled shared dev credentials)
2. System prompts: "Keep these settings or change to your own server?"
3. You write application code normally
4. Cursor AI detects when Temporal would help
5. Complete implementation appears automatically with tests
6. You review and use the generated code

---

## For Cursor AI: System Architecture

**Operating Mode:** AUTONOMOUS IMPLEMENTATION SYSTEM

**Execution Protocol:**
1. Read `summary.md` at session start (MANDATORY)
2. Validate `.env` configuration (AUTOMATIC)
3. Detect code patterns requiring Temporal (PROACTIVE)
4. Implement complete solutions (IMMEDIATE, NO PROMPTING)
5. Update `summary.md` with changes (AUTOMATIC)

**Reading Order (STRICT SEQUENCE - ALL PATHS DISCOVERED DYNAMICALLY):**
1. **Discover** `cursor-skills-temporal` folder by searching for fixed file names:
   - Search codebase for: `**/for-cursor/summary.md` (file name never changes)
   - Fallback: Search for: `**/progress/temporal-registry.json` (file name never changes)
   - Extract base path from found file
2. `{DISCOVERED_PATH}/for-cursor/summary.md` - System state (READ FIRST, ALWAYS - use discovered path, skip if missing)
3. `{DISCOVERED_PATH}/progress/temporal-registry.json` - Existing implementations (READ IMMEDIATELY AFTER SUMMARY - use discovered path, skip if missing)
4. Configuration validation - Check `.env` file (AUTOMATIC - at repo root)
5. `{DISCOVERED_PATH}/for-cursor/temporal-use-cases.mdc` - Pattern matching (use discovered path, skip if missing)
6. `{DISCOVERED_PATH}/for-cursor/temporal-workflows.mdc` + `temporal-activities.mdc` - Implementation patterns (use discovered path, skip if missing)
7. `{DISCOVERED_PATH}/for-cursor/temporal-language-examples.mdc` - Language-specific code (use discovered path, skip if missing)
8. Other `.mdc` files - Additional context as needed (use discovered path, skip if missing)

**Critical:** 
- All paths use dynamically discovered `cursor-skills-temporal` folder location
- File names are FIXED: `summary.md`, `temporal-registry.json`, `temporal-*.mdc` (never change)
- Folder names are FIXED: `for-cursor/`, `progress/`, `for-developer/` (never change)
- Only `cursor-skills-temporal/` folder location varies (can be anywhere in codebase)
- Never create new files - only read and update existing ones

The system supports three deployment phases: stage1 (development), stage2 (staging), and production, with environment-specific configuration management.

## System Architecture

### For Human Understanding

**Component Structure:**
```
.cursorrules                    ← Cursor AI behavior rules
summary.md                      ← System state tracker
.env                           ← Your Temporal server config
├── temporal-overview.mdc      ← Temporal platform concepts
├── temporal-workflows.mdc     ← Workflow patterns
├── temporal-activities.mdc    ← Activity patterns
├── temporal-workers.mdc       ← Worker deployment
├── temporal-testing.mdc       ← Testing strategies
├── temporal-deployment.mdc    ← Production deployment
├── temporal-use-cases.mdc     ← Use case detection
├── temporal-error-handling.mdc ← Error handling
└── temporal-language-examples.mdc ← Multi-language code
```

**How Components Work Together:**
1. `.cursorrules` tells Cursor how to operate autonomously
2. `summary.md` maintains state across chat sessions
3. `.mdc` files contain Temporal implementation knowledge
4. `.env` contains your Temporal server configuration

### For Cursor AI: Execution Flow

**Automatic Operation Flow (ALL PATHS DISCOVERED DYNAMICALLY):**
1. **Discovery:** Discover `cursor-skills-temporal` folder location → Store base path
2. **Session Start:** Read `{DISCOVERED_PATH}/for-cursor/summary.md` → Load system state (use discovered path)
3. **Registry Check:** Read `{DISCOVERED_PATH}/progress/temporal-registry.json` → Check existing implementations (use discovered path)
4. **Configuration Check:** Validate `.env` (at repo root) → Alert if incomplete
5. **Pattern Detection:** Monitor code → Identify Temporal opportunities
6. **Duplicate Check:** Verify registry → Avoid duplicate implementations
7. **Immediate Implementation:** Generate complete solution → No prompting
8. **Environment Integration:** Use `.env` configuration → Automatic
9. **Registry Update:** Update `{DISCOVERED_PATH}/progress/temporal-registry.json` → Track new implementation (use discovered path)
10. **State Update:** Modify `{DISCOVERED_PATH}/for-cursor/summary.md` → Document changes (use discovered path)

**Configuration Validation (AUTOMATIC):**
Before any implementation, check:
- Is `TEMPORAL_URI` configured? (not placeholder)
- Is `TEMPORAL_NAMESPACE` valid? (stage1/stage2/production)
- Are TLS certificates set if namespace=production?
- Are language dependencies installed?

If missing → Alert developer with specific instructions
If valid → Proceed with implementation silently

**Knowledge Base Reading Order (ALL PATHS DISCOVERED DYNAMICALLY):**
```
Priority 0: Discover cursor-skills-temporal folder location
Priority 1: {DISCOVERED_PATH}/for-cursor/summary.md (system state)
Priority 2: {DISCOVERED_PATH}/progress/temporal-registry.json (existing implementations)
Priority 3: .env validation (configuration check - at repo root)
Priority 4: {DISCOVERED_PATH}/for-cursor/temporal-use-cases.mdc (pattern matching)
Priority 5: {DISCOVERED_PATH}/for-cursor/temporal-workflows.mdc + temporal-activities.mdc (implementation)
Priority 6: {DISCOVERED_PATH}/for-cursor/temporal-language-examples.mdc (code generation)
Priority 7: Other .mdc files (additional context - use discovered path)
```

## Documentation Files

### System Operation Files (FOR CURSOR AI)
- **`.cursorrules`** - YOUR OPERATING SYSTEM - Cursor reads this AUTOMATICALLY at every session start (this is Cursor's default behavior for .cursorrules files)
- **`summary.md`** - YOUR STATE MEMORY - Read automatically after .cursorrules (instruction in .cursorrules)
- **`progress/temporal-registry.json`** - IMPLEMENTATION TRACKER - Tracks all workflows, activities, workers, clients (read at start, updated after implementations)
- **`SYSTEM_GUIDE.md`** - OPERATION MANUAL - Explains how automatic operation works (quick reference)

### Core Configuration (FOR DEVELOPERS)
- **`.cursorrules`** - Primary orchestration file that defines when and how Cursor AI should suggest Temporal implementations based on detected code patterns
- **`.env`** - Environment configuration file containing Temporal server URI, namespace, and connection credentials (must be created manually)
- **`.gitignore`** - Version control exclusions for sensitive files and build artifacts
- **`summary.md`** - System state tracker that maintains context across Cursor chat sessions, preventing invalid modifications after timeouts
- **`progress/temporal-registry.json`** - Implementation registry that tracks all Temporal workflows, activities, workers, and clients created in your codebase (automatically maintained by Cursor AI)

### Technical Documentation
- **`temporal-overview.mdc`** - Platform architecture, core concepts, service catalog, deployment options, and infrastructure requirements
- **`temporal-workflows.mdc`** - Workflow implementation patterns including sequential, parallel, saga, signal-based, and scheduled workflows with versioning strategies
- **`temporal-activities.mdc`** - Activity patterns covering retry logic, idempotency, heartbeats, cancellation, and external service integration
- **`temporal-workers.mdc`** - Worker configuration, deployment patterns for Docker/Kubernetes/serverless, health checks, and scaling strategies
- **`temporal-testing.mdc`** - Testing methodologies for unit tests, integration tests, and end-to-end testing with time manipulation
- **`temporal-deployment.mdc`** - Production deployment guides for Temporal Cloud, self-hosted Kubernetes, security configuration, monitoring, and disaster recovery
- **`temporal-error-handling.mdc`** - Error classification, retry policies, compensation patterns, timeout handling, and observability
- **`temporal-use-cases.mdc`** - Ten complete use case implementations including background jobs, distributed transactions, microservice orchestration, and data pipelines
- **`temporal-language-examples.mdc`** - Complete working examples in TypeScript, Python, Go, Java, .NET, PHP, and Ruby with project setup and configuration

## Environment Configuration

### Initial Setup

Create a `.env` file in the project root directory with the following required variables:

```
TEMPORAL_URI=<temporal-server-address>:7233
TEMPORAL_NAMESPACE=<stage1|stage2|production>
TASK_QUEUE=default
```

Optional variables for secure connections:
```
TEMPORAL_TLS_CERT_PATH=<path-to-client-certificate>
TEMPORAL_TLS_KEY_PATH=<path-to-client-key>
TEMPORAL_TLS_CA_PATH=<path-to-ca-certificate>
```

Worker configuration:
```
MAX_CONCURRENT_ACTIVITIES=100
MAX_CONCURRENT_WORKFLOWS=50
```

### Phase-Specific Configuration

**Stage1 (Development)**
```
TEMPORAL_URI=temporal-stage1.company.com:7233
TEMPORAL_NAMESPACE=stage1
```

**Stage2 (Staging)**
```
TEMPORAL_URI=temporal-stage2.company.com:7233
TEMPORAL_NAMESPACE=stage2
```

**Production**
```
TEMPORAL_URI=temporal-prod.company.com:7233
TEMPORAL_NAMESPACE=production
TEMPORAL_TLS_CERT_PATH=/etc/temporal/certs/client.pem
TEMPORAL_TLS_KEY_PATH=/etc/temporal/certs/client-key.pem
```

## Language Support

The knowledge base provides complete implementation examples for:

- TypeScript/JavaScript (Node.js)
- Python
- Go
- Java
- .NET (C#)
- PHP
- Ruby

Each language implementation includes:
- Project initialization and dependency management
- Configuration loading from environment variables
- Worker and client setup with TLS support
- Activity and workflow implementations
- Error handling and retry logic
- Graceful shutdown handling

## Setup and Operation

### One-Time Setup (For Developers)

**Step 1: Configure Environment (REQUIRED)**

Edit the `.env` file in the project root:

```env
# Replace placeholder values with your actual Temporal server details
TEMPORAL_URI=temporal.your-company.com:7233
TEMPORAL_NAMESPACE=stage1

# For production, also add:
# TEMPORAL_TLS_CERT_PATH=/path/to/client.pem
# TEMPORAL_TLS_KEY_PATH=/path/to/client-key.pem
```

**What Each Variable Means:**
- `TEMPORAL_URI` - Your company's Temporal server address with port
- `TEMPORAL_NAMESPACE` - Deployment phase (stage1=dev, stage2=staging, production=prod)
- `TASK_QUEUE` - Task queue name (default is fine for most cases)
- `TEMPORAL_TLS_CERT_PATH` - Certificate file for production (optional for dev)
- `TEMPORAL_TLS_KEY_PATH` - Private key for production (optional for dev)

**Step 2: Install Dependencies**

Choose your programming language:

**TypeScript/Node.js:**
```bash
npm install dotenv @temporalio/client @temporalio/worker @temporalio/workflow
```

**Python:**
```bash
pip install temporalio python-dotenv
```

**Go:**
```bash
go get github.com/joho/godotenv go.temporal.io/sdk
```

**Step 3: Verify Configuration**

Cursor AI will automatically check your configuration and alert you if anything is missing.

### Automatic Operation (For Developers)

After setup, the system works automatically:

1. **Write Code Normally** - Focus on your application logic
2. **Cursor Detects Patterns** - AI monitors for Temporal opportunities automatically
3. **Configuration Check** - System validates `.env` before implementing
4. **Implementation Appears** - Complete Temporal solution generated automatically
5. **Review and Use** - Code is production-ready with error handling and tests

**Example Developer Experience:**

```typescript
// You write this:
async function processOrder(orderId: string) {
  // Need to charge payment, update inventory, send email
  // This might fail and need retries
}

// Cursor automatically generates:
// - Complete workflow definition
// - Activity implementations
// - Worker setup code
// - Client code
// - Error handling with retry policies
// - Configuration loading from .env
// - Tests
```

**No Prompting Required:** If `.env` is configured, implementation happens automatically

**Configuration Feedback:** If `.env` is incomplete, Cursor tells you exactly what to configure

### Automatic Feedback System (For Cursor AI)

**Before Any Implementation:**

Check configuration and provide feedback:

```
✓ Configuration Valid:
  - TEMPORAL_URI: temporal-stage1.company.com:7233
  - TEMPORAL_NAMESPACE: stage1
  - TLS: Not required (development environment)
  
  Proceeding with automatic implementation...
```

or

```
⚠️ Configuration Required:
  - TEMPORAL_URI: Still set to localhost:7233
  - Please update .env with your company's Temporal server
  
  Required: Edit .env and set:
  TEMPORAL_URI=<your-temporal-server>:7233
  TEMPORAL_NAMESPACE=stage1
  
  After configuration, I'll automatically implement your workflow.
```

## Detection Patterns

Cursor AI suggests Temporal implementations when detecting:

- Long-running operations exceeding 30 seconds
- Multi-step business processes requiring coordination
- External API calls requiring retry logic
- Scheduled or recurring task execution
- Distributed transaction patterns (saga)
- State machine implementations
- Event-driven processing requirements
- Background job processing
- Human-in-the-loop approval workflows
- Data pipeline orchestration

## Security Considerations

All sensitive data is excluded from version control via `.gitignore`:
- Environment configuration files (.env)
- TLS certificates and private keys
- Build artifacts and dependencies

Production deployments require:
- TLS/mTLS for encrypted communication
- Certificate-based authentication
- Namespace isolation
- Encrypted data at rest using custom codecs

## Validation

Verify configuration before deployment:

**TypeScript:**
```bash
node -e "require('./src/config').config"
```

**Python:**
```bash
python -c "from config import config"
```

**Go:**
```bash
go run config/config.go
```

## Context Continuity System

### Chat Timeout Recovery
The system includes a context continuity mechanism to handle Cursor chat timeouts:

**summary.md File:**
- Tracks current system state and implementation progress
- Updated after each significant modification
- Contains validation checklists and completion status
- Lists prohibited modifications to prevent breaking changes
- Provides context for resuming work in new chat sessions

**Usage Protocol:**
1. Cursor AI reads `summary.md` at the start of each session
2. System state and pending work are loaded from summary
3. After completing tasks, `summary.md` is updated with changes
4. New chat sessions continue from documented state without data loss

**Benefits:**
- Prevents invalid modifications after chat timeouts
- Maintains implementation consistency across sessions
- Provides clear context on what has been completed
- Documents rationale for design decisions
- Tracks SDK versions and configuration status

## Support Resources

- Temporal UI available at `http://<TEMPORAL_URI>:8080`
- Documentation files contain language-specific troubleshooting
- Infrastructure team provides connection credentials and certificates
- `summary.md` provides current system state and context

## Understanding The System

### For Developers (Human Readers)

**What This System Does:**
This is an intelligent documentation system that teaches Cursor AI to write Temporal code automatically. Think of it as having an expert Temporal developer watching your code and implementing workflows proactively.

**What You Need to Know:**
1. Configure `.env` file once with your Temporal server details
2. Install language dependencies for your chosen language
3. Write code normally - Cursor handles Temporal automatically
4. Review generated implementations - they're production-ready

**Files You Interact With:**
- `.env` - Your configuration (edit this)
- Generated code - Review and use (Cursor creates this)

**Files You Don't Touch:**
- `.cursorrules` - Cursor AI behavior rules
- `summary.md` - System state tracker
- `temporal-*.mdc` - Knowledge base files

## For Cursor AI: How You Operate Automatically

**YOU HAVE AN AUTOMATIC OPERATING SYSTEM:**

The file `.cursorrules` is THE file that Cursor reads automatically at every session start. This is not optional - Cursor's architecture automatically reads `.cursorrules` to understand how to behave in this project.

**Automatic Reading Sequence (EVERY SESSION):**
```
Session Starts
    ↓
1. Cursor reads .cursorrules (AUTOMATIC - Cursor does this by default)
    ↓
2. .cursorrules tells you to read summary.md (AUTOMATIC - instruction in .cursorrules)
    ↓
3. .cursorrules tells you to validate .env (AUTOMATIC - instruction in .cursorrules)
    ↓
4. .cursorrules tells you to detect patterns (AUTOMATIC - instruction in .cursorrules)
    ↓
5. .cursorrules tells you to implement or feedback (AUTOMATIC - instruction in .cursorrules)
    ↓
6. .cursorrules tells you to update summary.md (AUTOMATIC - instruction in .cursorrules)
```

**Key Files in Automatic Operation:**
- **`.cursorrules`** → Cursor reads this AUTOMATICALLY (Cursor's default behavior)
- **`summary.md`** → You read this AUTOMATICALLY (because .cursorrules tells you to)
- **`SYSTEM_GUIDE.md`** → Quick reference for understanding automatic flow
- **`temporal-*.mdc`** → You reference these AUTOMATICALLY during implementation

**See `SYSTEM_GUIDE.md` for complete automatic operation documentation.**

**Critical Reading Order (AUTOMATIC - EVERY SESSION):**
1. ALWAYS read `summary.md` first (MANDATORY at session start)
2. ALWAYS perform COMPLETE `.env` validation (DYNAMIC and comprehensive)
3. ALWAYS update `summary.md` after changes (AUTOMATIC update)

**Configuration Validation Protocol (DYNAMIC):**
Before implementing, perform COMPLETE validation:
```python
# Pseudo-code for DYNAMIC validation
validation_results = {
    'missing': [],
    'invalid': [],
    'warnings': []
}

# Check ALL required variables (adapts to namespace)
for var in get_required_vars(namespace):
    if not exists(var):
        validation_results['missing'].append({
            'var': var,
            'solution': get_solution_for(var),
            'reason': get_reason_for(var)
        })

# Check language dependencies AND versions (adapts to detected language)
for package, min_version in get_required_packages_with_versions(detected_language):
    if not installed(package):
        validation_results['missing'].append({
            'package': package,
            'install_cmd': get_install_cmd(detected_language, min_version),
            'reason': f'{detected_language} requires {package}'
        })
    else:
        # Check version dynamically
        installed_version = get_installed_version(package, detected_language)
        if installed_version and compare_versions(installed_version, min_version) < 0:
            validation_results['outdated'].append({
                'package': package,
                'installed': installed_version,
                'required': min_version,
                'update_cmd': get_update_cmd(detected_language, package, min_version),
                'reason': f'{package} version {installed_version} is below minimum {min_version}'
            })

# Generate DYNAMIC feedback
if has_issues(validation_results):
    generate_complete_feedback(validation_results)  # Lists ALL issues
else:
    implement_silently()  # 100% valid, proceed
```

**Validation is DYNAMIC:**
- Not hardcoded checks
- Adapts to language, namespace, project structure
- Lists EVERY missing item
- Provides specific solutions for detected environment

If ANY issue found → Generate COMPLETE customized feedback
If 100% valid → Implement silently

**Implementation Protocol:**
- Detect pattern → Check config → Implement automatically
- No prompting, no confirmation, no questions
- Complete solutions only (workflows + activities + workers + tests)
- Update summary.md after completion

**File Reading Sequence:**
```
summary.md → .env validation → temporal-use-cases.mdc → 
temporal-workflows.mdc & temporal-activities.mdc →
temporal-language-examples.mdc → other .mdc files (as needed)
```

## Version Information

### SDK Version Checking (Dynamic - Not Hardcoded)

Cursor AI automatically checks SDK versions dynamically:

- **Automatically fetches latest versions from https://docs.temporal.io/** using web_search tool - not hardcoded
- **Detects installed versions** from package files (`package.json`, `requirements.txt`, `go.mod`, etc.)
- **Compares versions** and alerts if outdated
- **Alerts appear in Cursor chat** when validation runs

**Current Minimum SDK Versions (Auto-Fetched from https://docs.temporal.io/):**
- TypeScript: 1.10.0+ (Updates API, Nexus support)
- Python: 1.5.0+ (enhanced async, type hints)
- Go: 1.25.0+ (Nexus support, Updates API)
- Java: 1.23.0+ (Spring integration, Updates API)
- .NET: 1.1.0+ (async improvements)
- PHP: 2.10.0+ (PHP 8.2+ support)
- Ruby: 0.1.0+ (Preview/Beta)

**See:** 
- `VERSION_CHECKING_GUIDE.md` - Detailed information on version checking and alerts
- `MAINTENANCE_GUIDE.md` - Periodic maintenance tasks and updates

All examples follow Temporal best practices, including support for Updates, Nexus operations, and enhanced observability features.

