# Temporal Integration Scripts - Usage Guide

**Last Updated:** January 21, 2026  
**System Version:** 2.0

---

## 📋 Available Scripts

### **Essential Scripts (Used in Automated Workflow)**

| Script | Purpose | Usage | Context |
|--------|---------|-------|---------|
| `setup-temporal.py` | Deploy skill to module | Manual/Agent | Setup phase |
| `scan-codebase.py` | Scan APIs (Phase 1 & 2) | Automatic | Scanning |
| `deep-scanner.py` | Deep call chain analysis | Automatic | Scanning |
| `priority-analyzer.py` | Assign priority scores | Automatic | Scanning |
| `detect-mode.py` | Detect automation mode | Automatic | Mode detection |
| `refactor-to-temporal.py` | Implement Temporal code | Automatic | Implementation |
| `generate-tests.py` | Generate unit tests | Automatic | Testing |
| `run-tests.py` | Run generated tests | Automatic | Testing |
| `create-test-reports.py` | Create test reports | Automatic | Reporting |
| `validate-system.py` | Validate configuration | Automatic | Validation |
| `cleanup-code.py` | Remove AI slop | Automatic | Post-processing |

---

## 🚀 Script Details

### **1. setup-temporal.py** (Deploy Temporal Skill)

**Purpose:** Deploy Temporal skill to a target module

**Usage:**
```bash
# Deploy to module
python setup-temporal.py /path/to/target_module

# With custom source
python setup-temporal.py /path/to/target_module /path/to/skills
```

**What it does:**
1. Copies `skills/temporal/` folder to target module
2. Merges temporal `.cursorrules` with existing (or creates new)
3. Creates backup of original `.cursorrules` (if exists)
4. Initializes registry (`progress/temporal-registry.json`)

**When to use:**
- **Agent:** First step when deploying to new module
- **Developer:** When manually adding Temporal to existing module

**Example:**
```bash
# From template repo
cd cursor-skills-template
python skills/temporal/scripts/setup-temporal.py ../m46_FTP_java

# Output:
# ✅ Copied skills/temporal/ to target
# ✅ Merged .cursorrules (backup created)
# ✅ Initialized registry
```

---

### **2. scan-codebase.py** (Two-Phase Scanner)

**Purpose:** Scan codebase for APIs and services

**Usage:**
```bash
# Phase 1: Scan APIs only (fast)
python scan-codebase.py --phase 1

# Phase 2: Scan services for selected APIs (targeted)
python scan-codebase.py --phase 2 --selected-apis 1,2,5

# Agent mode (auto-select HIGH priority)
python scan-codebase.py --phase 1 --agent-mode
python scan-codebase.py --phase 2 --selected-apis HIGH --agent-mode
```

**What it does:**
- **Phase 1:** Finds all internal/external APIs, assigns priorities
- **Phase 2:** Finds services using selected APIs, traces call chains

**Outputs:**
- `/tmp/scan-results-phase1.json` (Phase 1 results)
- `/tmp/scan-results-phase2.json` (Phase 2 results)

---

### **3. deep-scanner.py** (Call Chain Tracer)

**Purpose:** Deep analysis of API usage and call chains

**Usage:**
```bash
# Called automatically by scan-codebase.py Phase 2
# Can also be run standalone for analysis
python deep-scanner.py
```

**What it does:**
- Traces call chains up to 5 levels deep
- Maps dependencies (Controller → Service → Repository)
- Detects async usage (@Async methods)
- Cross-folder and cross-file tracing

---

### **4. priority-analyzer.py** (Priority Scoring)

**Purpose:** Assign HIGH/LOW priority scores to APIs

**Usage:**
```bash
# Called automatically by scan-codebase.py Phase 1
python priority-analyzer.py --input /tmp/scan-results-phase1.json
```

**What it does:**
- Analyzes complexity of each API
- Assigns priority score (0-100)
- Groups APIs into HIGH (>= 50) and LOW (< 50)

**Outputs:**
- `/tmp/scan-results-phase1-prioritized.json`

---

### **5. detect-mode.py** (Mode Detection)

**Purpose:** Detect which automation mode to use

**Usage:**
```bash
# Called automatically by .cursorrules
python detect-mode.py
```

**What it does:**
- Checks if code exists in repo
- Checks if registry exists
- Checks if `processed_by_agent: true`

**Returns:**
- `new-module` - No code yet
- `post-agent` - Agent already processed
- `first-time-existing` - Code exists, no registry

---

### **6. refactor-to-temporal.py** (Implementation Engine)

**Purpose:** Implement Temporal workflows, activities, workers

**Usage:**
```bash
# Implement with strategy
python refactor-to-temporal.py \
  --selected-apis "1,2,5" \
  --strategy union \
  --activity-type shared

# Update registry after implementation
python refactor-to-temporal.py --update-registry

# Agent mode
python refactor-to-temporal.py --agent-mode
```

**What it does:**
- Generates Temporal workflows
- Generates Temporal activities
- Generates Temporal workers
- Creates temporal/ folder structure

**Outputs:**
- `temporal/workflows/*.{java,py,ts,...}`
- `temporal/activities/*.{java,py,ts,...}`
- `temporal/workers/*.{java,py,ts,...}`
- `/tmp/implementation-manifest.json`

---

### **7. generate-tests.py** (Test Generator)

**Purpose:** Generate unit tests for Temporal implementations

**Usage:**
```bash
# Generate tests from manifest
python generate-tests.py --manifest /tmp/implementation-manifest.json
```

**What it does:**
- Generates workflow tests
- Generates activity tests
- Generates worker tests
- Generates integration tests

**Outputs:**
- `temporal/tests/workflows/test_*.{java,py,ts,...}`
- `temporal/tests/activities/test_*.{java,py,ts,...}`
- `temporal/tests/workers/test_*.{java,py,ts,...}`
- `temporal/tests/integration/test_*.{java,py,ts,...}`
- `/tmp/test-manifest.json`

---

### **8. run-tests.py** (Test Executor)

**Purpose:** Run generated tests and capture results

**Usage:**
```bash
# Run tests from manifest
python run-tests.py --test-manifest /tmp/test-manifest.json
```

**What it does:**
- Executes all generated tests (pytest/jest/go test/etc.)
- Captures pass/fail results
- Calculates coverage

**Outputs:**
- `/tmp/test-results.json`

---

### **9. create-test-reports.py** (Report Generator)

**Purpose:** Create markdown test reports

**Usage:**
```bash
# Create reports from results
python create-test-reports.py \
  --results /tmp/test-results.json \
  --manifest /tmp/implementation-manifest.json
```

**What it does:**
- Generates detailed REPORT.md
- Generates summary SUMMARY.md

**Outputs:**
- `temporal/tests/REPORT.md`
- `temporal/tests/SUMMARY.md`

---

### **10. validate-system.py** (Configuration Validator)

**Purpose:** Validate configuration before implementing

**Usage:**
```bash
# Validate current configuration
python validate-system.py
```

**What it does:**
- Checks `.env` file exists
- Validates required variables
- Checks dependencies installed
- Validates SDK versions (fetches from docs.temporal.io)

**Returns:**
- Exit code 0 if valid
- Exit code 1 if invalid (with detailed feedback)

---

### **11. cleanup-code.py** (Code Cleanup)

**Purpose:** Remove emojis, images, AI-generated slop

**Usage:**
```bash
# Cleanup with backup
python cleanup-code.py --backup-dir .temporal-backups

# Cleanup without backup
python cleanup-code.py
```

**What it does:**
- Creates backup (if specified)
- Removes emojis from code files
- Removes images/base64 data
- Validates cleaned code

**Outputs:**
- Cleaned code files (in-place)
- Backup in `.temporal-backups/` (if specified)

---

## 🔄 Typical Workflow (Automated)

### **Agent Workflow:**
```bash
1. setup-temporal.py           # Deploy to module
2. scan-codebase.py --phase 1  # Find APIs
3. priority-analyzer.py        # Assign priorities
4. scan-codebase.py --phase 2  # Find services (HIGH only)
5. deep-scanner.py             # Trace call chains
6. validate-system.py          # Check config
7. refactor-to-temporal.py     # Implement
8. generate-tests.py           # Generate tests
9. run-tests.py                # Run tests
10. create-test-reports.py     # Create reports
11. cleanup-code.py            # Clean code
12. refactor-to-temporal.py --update-registry  # Update registry
```

### **Developer Workflow (Interactive):**
```bash
1. setup-temporal.py           # Manual: Deploy to module
2. Open Cursor IDE             # Cursor reads .cursorrules at root
3. System prompts              # Interactive: "Scan codebase? [Y/N]"
4. scan-codebase.py --phase 1  # User approves
5. System prompts              # Interactive: "Choose APIs"
6. scan-codebase.py --phase 2  # After user selects
7. System prompts              # Interactive: "Approve implementation?"
8. (Steps 6-12 above)          # After user approves
```

---

## 📂 File Locations

**Scripts:** `skills/temporal/scripts/`  
**Temp Files:** `/tmp/` (deleted after use)  
**Outputs:** `temporal/` (in target module)  
**Registry:** `skills/temporal/progress/temporal-registry.json`

---

## 🚨 Notes

- **All scripts support `--help`** for detailed usage
- **Temp files in `/tmp/`** are automatically cleaned up
- **Scripts are language-agnostic** (detect language automatically)
- **Registry must be updated** after each implementation (prevents duplicates)

---

**Last Updated:** January 21, 2026  
**System Version:** 2.0
