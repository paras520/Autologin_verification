# Dynamic Folder Discovery Protocol

## Overview

The `cursor-skills-temporal` folder can be placed **anywhere** in a repository. This protocol defines how Cursor AI automatically discovers its location at every session start.

## Why Dynamic Discovery?

**Flexibility:** Developers can place the `cursor-skills-temporal` folder anywhere in their codebase:
- Root level: `/cursor-skills-temporal/`
- In a tools directory: `/tools/cursor-skills-temporal/`
- In documentation: `/docs/cursor-skills-temporal/`
- Nested anywhere: `/some/deep/path/cursor-skills-temporal/`

**The system finds it automatically** - no configuration needed.

---

## Discovery Method

**BEFORE reading any files, you MUST dynamically discover the `cursor-skills-temporal` folder location:**

### Search by File Names (They Never Change)

**Step 1: Primary Search**
- Search for `summary.md` file inside `for-cursor/` folder anywhere in codebase
- File name: `summary.md` (never changes)
- Folder: `for-cursor/` (never changes)
- **Search pattern:** `**/for-cursor/summary.md`

**Step 2: Fallback Search**
- If not found, search for `temporal-registry.json` inside `progress/` folder
- File name: `temporal-registry.json` (never changes)
- Folder: `progress/` (never changes)
- **Search pattern:** `**/progress/temporal-registry.json`

**Step 3: Store Base Path**
- Once found, extract the base path (the `cursor-skills-temporal/` folder containing these subfolders)
- **Example:** If found `some/path/cursor-skills-temporal/for-cursor/summary.md`
- **Base path** = `some/path/cursor-skills-temporal/`
- Store as `{BASE_PATH}` or `{DISCOVERED_PATH}` for all file operations

**Step 4: Use Dynamic Paths**
- All file references use this discovered base path:
  - `{BASE_PATH}/for-cursor/summary.md`
  - `{BASE_PATH}/progress/temporal-registry.json`
  - `{BASE_PATH}/for-cursor/temporal-*.mdc`
  - `{BASE_PATH}/for-cursor/references/*.md`
  - `{BASE_PATH}/for-developer/*.md`

---

## Fixed Names (Never Change)

### Files
- `summary.md` - System state tracker
- `temporal-registry.json` - Implementation registry
- `temporal-*.mdc` - Knowledge base files (9 files with fixed pattern)
- `discovery-protocol.md`, `execution-protocol.md`, etc. - Reference files

### Folders
- `for-cursor/` - AI knowledge base
- `for-cursor/references/` - Protocol reference files
- `progress/` - Implementation tracking
- `for-developer/` - Human documentation
- `scripts/` - Automation scripts

**Critical:** Only the location of `cursor-skills-temporal/` folder varies (can be anywhere in codebase). Everything inside has fixed names.

---

## Error Handling for File Reading

**If files cannot be found:**
- Skip missing files and continue (don't get stuck)
- If `summary.md` is missing, skip it and continue (don't create new files)
- If `temporal-registry.json` is missing, skip it and continue (don't create new files)
- If any `.mdc` file cannot be read, skip it and use available knowledge
- If any reference file missing, continue with available protocols

**Never:**
- Create new files - only update existing ones
- Fail completely due to one missing file - always continue operation
- Get stuck waiting for a file that doesn't exist

**Always:**
- Continue with available knowledge
- Log errors internally
- Proceed with implementation using available files

---

## Why This Approach Works

1. **File names are fixed** - More reliable than searching for folder paths
2. **Multiple fallbacks** - If one search fails, try another
3. **Folder agnostic** - Works anywhere in repository structure
4. **No configuration** - Automatic discovery, no setup needed
5. **Robust** - Handles missing files gracefully

---

## Implementation Example

```python
# Pseudo-code for discovery
def discover_base_path():
    # Primary search
    summary_path = search_codebase("**/for-cursor/summary.md")
    if summary_path:
        return extract_base_path(summary_path)
    
    # Fallback search
    registry_path = search_codebase("**/progress/temporal-registry.json")
    if registry_path:
        return extract_base_path(registry_path)
    
    # Discovery failed
    return None

def extract_base_path(file_path):
    # Extract everything up to and including 'cursor-skills-temporal/'
    # Example: "some/path/cursor-skills-temporal/for-cursor/summary.md"
    # Returns: "some/path/cursor-skills-temporal/"
    return path_up_to_folder(file_path, "cursor-skills-temporal")
```

---

## Usage in File Operations

Once discovered, ALL file operations use the base path:

```python
# After discovery
BASE_PATH = discover_base_path()  # e.g., "tools/cursor-skills-temporal/"

# All file reads use discovered path
read_file(f"{BASE_PATH}/for-cursor/summary.md")
read_file(f"{BASE_PATH}/progress/temporal-registry.json")
read_file(f"{BASE_PATH}/for-cursor/temporal-workflows.mdc")
read_file(f"{BASE_PATH}/for-cursor/references/execution-protocol.md")

# All file writes use discovered path
write_file(f"{BASE_PATH}/for-cursor/summary.md", updated_content)
write_file(f"{BASE_PATH}/progress/temporal-registry.json", updated_registry)
```

**Never hardcode paths** - always use dynamic discovery.

---

## Priority: Execute First

This discovery protocol **MUST execute BEFORE any other file operations**. It is Priority 0 in the reading order.

**Sequence:**
1. **Discovery Protocol** (this file) - Find folder location
2. Load system state from `summary.md`
3. Load registry from `temporal-registry.json`
4. **NEW: Scan entire codebase** (run `{DISCOVERED_PATH}/scripts/scan-codebase.py`)
5. **NEW: Detect Temporal implementations** (check scan results)
6. **NEW: Detect patterns in existing code** (from scan results)
7. **NEW: Analyze Temporal efficiency** (if Temporal found, run `{DISCOVERED_PATH}/scripts/analyze-temporal.py`)
8. Validate configuration
9. Read knowledge base files as needed

---

## New Discovery Steps (Modular Functionality)

### Step 5: Codebase Scanning
**Action:** Run `{DISCOVERED_PATH}/scripts/scan-codebase.py` to scan entire repository
**Purpose:** Discover all code files, detect Temporal implementations, detect patterns
**Result:** Scan report with Temporal detection and pattern analysis

### Step 6: Temporal Detection
**Action:** Check scan report for Temporal implementations
**Checks:**
- Temporal SDK imports
- Workflow definitions
- Activity definitions
- Worker setups
**Result:** List of existing Temporal services or "none found"

### Step 7: Pattern Detection (Existing Code)
**Action:** Analyze scan report for patterns suitable for Temporal
**Patterns:**
- Long-running operations
- Multi-step processes
- External API calls
- Background jobs
- Scheduled tasks
**Result:** List of patterns found in existing code

### Step 8: Efficiency Analysis (If Temporal Found)
**Action:** Run `{DISCOVERED_PATH}/scripts/analyze-temporal.py` with scan report
**Purpose:** Analyze existing Temporal code for efficiency
**Checks:**
- Retry policies
- Error handling
- Activity organization
- Workflow structure
**Result:** Efficiency analysis with improvement suggestions

---

**Last Updated:** January 13, 2026 (Added modular functionality steps)

