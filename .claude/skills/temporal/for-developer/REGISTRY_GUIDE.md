# Temporal Implementation Registry Guide

## What Is The Registry?

The `temporal-registry.json` file is an automatic tracking system that maintains a complete record of all Temporal implementations in your codebase. Cursor AI updates this file automatically whenever it creates Temporal workflows, activities, workers, or clients.

## Purpose

**For Developers:**
- See what Temporal components exist in your codebase
- Track implementation statistics
- Verify Temporal is being used
- Understand project structure

**For Cursor AI:**
- Avoid duplicate implementations
- Track what has been created
- Maintain consistency across sessions
- Reference existing code

## File Location

The registry file is located at:
```
{repository-root}/cursor-skills-temporal/progress/temporal-registry.json
```

**Path Discovery:** Cursor AI automatically discovers the `cursor-skills-temporal` folder location relative to the repository root (where `.cursorrules` is located). The folder can be placed anywhere in your repository.

**Note:** This file is automatically maintained by Cursor AI. You can read it, but don't manually edit it - Cursor will update it automatically.

## Registry Structure

The registry contains:

### 1. Implementations Section
Lists all Temporal components created:

- **Workflows:** All workflow definitions
- **Activities:** All activity implementations
- **Workers:** All worker configurations
- **Clients:** All client implementations

### 2. Files Section
Tracks file locations:

- **workflowFiles:** Paths to workflow files
- **activityFiles:** Paths to activity files
- **workerFiles:** Paths to worker files
- **clientFiles:** Paths to client files
- **configFiles:** Paths to configuration files

### 3. Statistics Section
Aggregated data:

- **totalWorkflows:** Count of workflows
- **totalActivities:** Count of activities
- **totalWorkers:** Count of workers
- **totalClients:** Count of clients
- **languages:** Language usage breakdown
- **taskQueues:** List of task queues used

### 4. Metadata Section
Project-level information:

- **projectLanguage:** Primary language detected
- **temporalNamespace:** Namespace being used
- **temporalUri:** Temporal server URI
- **firstImplementation:** Timestamp of first implementation
- **lastImplementation:** Timestamp of most recent implementation

## Example Registry Entry

```json
{
  "version": "1.0.0",
  "lastUpdated": null,
  "implementations": {
    "workflows": [
      {
        "name": "processOrder",
        "file": "src/workflows/order.ts",
        "language": "typescript",
        "taskQueue": "order-processing",
        "description": "Processes order payment and fulfillment",
        "created": null,
        "status": "active"
      }
    ],
    "activities": [
      {
        "name": "chargePayment",
        "file": "src/activities/payment.ts",
        "language": "typescript",
        "description": "Charges customer payment",
        "created": null,
        "status": "active"
      }
    ]
  },
  "statistics": {
    "totalWorkflows": 1,
    "totalActivities": 1,
    "totalWorkers": 0,
    "totalClients": 0,
    "languages": {
      "typescript": 2
    },
    "taskQueues": ["order-processing"]
  },
  "metadata": {
    "projectLanguage": "typescript",
    "temporalNamespace": "stage1",
    "temporalUri": "temporal-stage1.company.com:7233",
    "firstImplementation": null,
    "lastImplementation": null
  }
}
```

## How It Works

### Automatic Updates (Fully Automatic - No Developer Interaction)

Cursor AI automatically:
1. **Discovers** `cursor-skills-temporal` folder by searching for fixed file names
2. **Reads** the registry at session start (using discovered path)
3. **Checks** for duplicates before implementing (prevents duplicate code)
4. **Updates** the registry after every implementation (automatic tracking)
5. **Maintains** accurate statistics and metadata (automatic)
6. **Updates** summary.md after every change (automatic state tracking)
7. **Preserves** context across chat sessions (automatic continuity)

**No developer prompts or confirmations needed - everything happens automatically.**

### When Registry Is Updated

The registry is updated automatically when Cursor AI:
- Creates a new workflow
- Creates a new activity
- Creates a new worker
- Creates a new client
- Modifies existing Temporal code

### What Gets Tracked

For each implementation:
- **Name:** Component name
- **File Path:** Location in codebase
- **Language:** Programming language used
- **Description:** What the component does
- **Created Timestamp:** When it was created
- **Status:** active, deprecated, etc.
- **Task Queue:** (for workflows) Which queue it uses

## Reading The Registry

### Check What Exists

```bash
# View registry (JSON format)
cat cursor-skills-temporal/progress/temporal-registry.json

# Pretty print (requires jq)
cat cursor-skills-temporal/progress/temporal-registry.json | jq
```

### Common Queries

**Count workflows:**
```bash
jq '.statistics.totalWorkflows' cursor-skills-temporal/progress/temporal-registry.json
```

**List all workflow files:**
```bash
jq '.files.workflowFiles[]' cursor-skills-temporal/progress/temporal-registry.json
```

**See language breakdown:**
```bash
jq '.statistics.languages' cursor-skills-temporal/progress/temporal-registry.json
```

**Find all implementations:**
```bash
jq '.implementations' cursor-skills-temporal/progress/temporal-registry.json
```

## Benefits

### For Development

1. **Visibility:** See all Temporal code at a glance
2. **Tracking:** Monitor implementation progress
3. **Verification:** Confirm Temporal is being used
4. **Documentation:** Registry serves as implementation catalog

### For Cursor AI

1. **Duplicate Prevention:** Avoids creating duplicate workflows/activities
2. **Context Awareness:** Knows what already exists
3. **Consistency:** Maintains accurate project state
4. **Continuity:** Preserves knowledge across sessions

## Important Notes

### Don't Manually Edit

The registry is automatically maintained by Cursor AI. Manual edits may cause:
- Inconsistencies
- Duplicate tracking issues
- Statistics errors
- Session continuity problems

### If Registry Gets Out of Sync

If the registry doesn't match your codebase:
1. Cursor will detect discrepancies
2. Registry will be updated automatically
3. No manual intervention needed

### Registry Location

The registry is in `progress/` folder because:
- It tracks implementation progress (useful for both developers and Cursor)
- It's separate from knowledge base files
- It's easier to find and reference
- Developers can read it to see what's been implemented

## Troubleshooting

### Registry Not Updating

If registry isn't updating:
1. Check `.cursorrules` is at repo root
2. Verify Cursor is reading `.cursorrules`
3. Check file permissions on registry
4. Ensure Cursor has write access

### Registry Shows Wrong Data

If registry shows incorrect information:
1. Cursor will auto-correct on next implementation
2. Registry updates happen automatically
3. No manual fixes needed

### Missing Implementations

If implementations are missing from registry:
1. Cursor will add them on next session
2. Registry scans codebase automatically
3. Updates happen during implementation

## Summary

The registry is an automatic tracking system that:
- Maintains a complete record of Temporal implementations
- Prevents duplicate code creation
- Provides visibility into project structure
- Enables Cursor AI to maintain context

**You don't need to do anything** - Cursor AI handles registry updates automatically. Just read it when you want to see what Temporal code exists in your project.

