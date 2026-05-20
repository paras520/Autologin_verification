# Implementation Guidelines

## Overview

This protocol defines how Cursor AI generates Temporal implementations. These guidelines ensure consistent, production-ready, best-practice code generation.

---

## Implementation Style

**PROACTIVE:** Implement before developer asks (if config valid)
**VALIDATED:** Check configuration before implementing  
**COMPLETE:** Full workflows, activities, workers, tests, deployment
**AUTOMATIC:** Use `.env` config without asking
**SILENT:** No "should I implement this?" questions (unless config missing)
**FEEDBACK:** Alert developer if configuration incomplete

---

## Two-Phase Implementation Strategy

The implementation follows a two-phase workflow for better developer control:

**Phase 1: API Selection**
1. Scan codebase for APIs (internal + external) - FAST
2. Present APIs to developer with priorities
3. Wait for developer to select which APIs to implement
4. Proceed to Phase 2 with selected APIs

**Phase 2: Service Implementation**
1. Scan for services/functions using selected APIs - TARGETED
2. Present services report to developer
3. Ask implementation strategies:
   - Multiple APIs: Union vs Intersection
   - Each API: Shared Activity vs Per-Service Activities
4. Wait for developer approval
5. Implement in temporal/ folder structure
6. Update registry and continue monitoring

**Implementation Strategies:**

**For Multiple APIs:**
- **Union:** Implement services using ANY of the selected APIs (more inclusive)
- **Intersection:** Implement ONLY services using ALL selected APIs (more restrictive)

**For API Activities:**
- **Shared Activity:** One reusable activity for all services (recommended for consistency)
- **Per-Service Activities:** Separate activity per service (recommended for isolation)

---

## Priority Scoring System (NEW - Phase 2)

**Purpose:** Automatically classify APIs/services as HIGH or LOW priority for Temporal implementation

**Scoring Algorithm (0-100):**
- +30 points: Has external API calls (OpenAI, Stripe, AWS, payment, email, SMS, etc.)
- +25 points: Long-running operation (sleep, batch processing, migrations, etc.)
- +20 points: Multi-step workflow (phases, pipelines, orchestration)
- +15 points: Needs retry logic (error-prone operations like payments, bookings)
- +10 points: Complex state management (database, cache, session state)

**Priority Classification:**
- **HIGH priority (≥50)**: Strongly recommended for Temporal (external APIs, complex workflows)
- **LOW priority (<50)**: Simple operations that may not benefit from Temporal (health checks, CRUD)

**Display in Phase 1:**
- APIs sorted by priority (HIGH first)
- Show score, priority level, and explanation
- Recommend implementing HIGH priority items first

---

## Automation Behavior (NEW - Phase 2)

**Three Automation Modes:**

**Mode 1: NEW MODULE**
- **Trigger:** No significant code yet OR no registry
- **Behavior:** Auto-implement everything immediately
- **No approval needed:** Fully automatic with notifications
- **Use case:** New projects, developers writing fresh code

**Mode 2: POST-AGENT**
- **Trigger:** Registry exists with `processed_by_agent: true`
- **Behavior:** Show status, ask once for remaining LOW priority items
- **Agent report:** "✅ Implemented X HIGH priority, ⏭️ Skipped Y LOW priority"
- **Ask once:** "Would you like to implement any skipped items?"
- **Future code:** Auto-implement with notifications
- **Use case:** After GitHub org-wide agent processes repo

**Mode 3: FIRST-TIME EXISTING**
- **Trigger:** Code exists but no registry
- **Behavior:** Ask once with priorities, then auto-implement forever
- **Initial:** Run Phase 1+2 with priority scores, ask which to implement
- **Registry created:** Marks module as "Temporal-enabled"
- **Future code:** Auto-implement with notifications
- **Use case:** Existing projects being introduced to Temporal

**Key Principle:** Ask once per module, then auto-implement ALL future code

---

## Automatic Implementation Strategy

### Step 1: Silent Detection & Registry Check

When implementing selected APIs/services:

1. **Analyze Code** - Understand intent and requirements automatically
2. **Reference Use Cases** - Check `temporal-use-cases.mdc` to determine best pattern
3. **Check Registry** - Verify `temporal-registry.json` for existing implementations
4. **Duplicate Check** - If similar exists, reference it instead of creating new
5. **Apply Strategy** - Use developer-selected implementation strategy (shared vs per-service)

### Step 2: Immediate Implementation

AUTOMATICALLY generate:
- Complete workflow definitions
- Activity implementations
- Worker setup code
- Client code for starting workflows
- Configuration using `.env` variables
- Error handling with retry policies
- All required imports and dependencies

### Step 3: Apply Best Practices

Follow these principles:
- Reference official patterns from `temporal-workflows.mdc` and `temporal-activities.mdc`
- Use language-appropriate idioms from `temporal-language-examples.mdc`
- Apply error handling patterns from `temporal-error-handling.mdc`
- Include testing strategies from `temporal-testing.mdc`
- Add deployment guidance from `temporal-deployment.mdc`

### Step 4: Generate Complete Solution

AUTOMATICALLY include (no prompting):
- Workflow definitions
- Activity implementations
- Worker setup code
- Client code for starting workflows
- Configuration files (temporal.yaml, docker-compose, etc.)
- Error handling and retry policies
- Logging and observability hooks
- Unit and integration tests
- README with setup instructions

### Step 5: Post-Implementation Cleanup (NEW)

AFTER generating code, AUTOMATICALLY:
- Create backup of generated files to `.temporal-backups/`
- Run cleanup script (`scripts/cleanup-code.py`) to remove emojis/images
- Validate cleaned files (check files exist, readable, no errors)
- If cleanup succeeds: Proceed to registry update
- If cleanup fails: Restore backup, skip registry update, alert developer

### Step 6: Update Registry (CONDITIONAL - Only if cleanup succeeds)

After implementing ANY Temporal component AND cleanup succeeds, AUTOMATICALLY update `{DISCOVERED_PATH}/progress/temporal-registry.json`:

**Tracking Protocol:**
1. **Workflows:** Add with name, file path, language, task queue, description, timestamp
2. **Activities:** Add with name, file path, language, description, timestamp
3. **Workers:** Add with file path, language, task queues, timestamp
4. **Clients:** Add with file path, language, timestamp
5. **Statistics:** Update counters, language counts, task queues
6. **Metadata:** Set project language, namespace, URI if first implementation
7. **Timestamps:** Update `lastUpdated`, `lastImplementation`

---

## File Organization: temporal/ Folder Structure (CRITICAL)

**ALL Temporal implementation files MUST be created inside `temporal/` folder at the repository root.**

### Required Folder Structure

```
temporal/
├── workflows/              # All workflow definitions
│   ├── [service]_workflow.py
│   ├── [service]_workflow.ts
│   └── ...
│
├── activities/            # All activity implementations
│   ├── [api_name]_activity.py
│   ├── [service]_[function]_activity.py
│   ├── shared/           # Shared/reusable activities
│   │   └── common_activities.py
│   └── ...
│
├── workers/              # Worker configuration files
│   ├── main_worker.py
│   ├── worker_config.py
│   └── ...
│
├── client/               # Client initialization and starters
│   ├── temporal_client.py
│   ├── workflow_starters.py
│   └── ...
│
├── tests/                # All tests
│   ├── workflows/       # Workflow tests
│   │   ├── test_[workflow].py
│   │   └── ...
│   ├── activities/      # Activity tests
│   │   ├── test_[activity].py
│   │   └── ...
│   └── integration/     # Integration tests
│       └── test_end_to_end.py
│
├── config/               # Configuration files
│   ├── temporal_config.py
│   ├── retry_policies.py
│   └── ...
│
└── README.md            # Usage documentation
```

### Naming Conventions

**For API-Based Implementation:**
- Activity file: `[api_name]_activity.[ext]`
  - Example: `openai_api_activity.py`
  - Example: `stripe_api_activity.ts`
- Sanitize API names: Replace spaces and special chars with underscores
  - "OpenAI API" → `openai_api_activity.py`
  - "Claude API (Anthropic)" → `claude_api_anthropic_activity.py`

**For Service-Based Implementation:**
- Workflow file: `[service_name]_workflow.[ext]`
  - Example: `payment_processing_workflow.py`
  - Example: `order_fulfillment_workflow.ts`
- Activity file: `[service]_[function]_activity.[ext]`
  - Example: `payment_capture_payment_activity.py`
  - Example: `order_validate_inventory_activity.ts`

**For Shared/Reusable Components:**
- Place in `activities/shared/` folder
- Use descriptive names: `http_client_activity.py`, `database_activity.py`

### File Creation Rules

**CRITICAL RULES:**
1. **NEVER create Temporal files in root directory**
2. **NEVER create Temporal files scattered across project**
3. **ALWAYS use temporal/ folder structure**
4. **ALWAYS create subdirectories if they don't exist**
5. **ALWAYS generate README.md in temporal/ folder**

### Implementation Type Mappings

**API Implementation:**
- Files go in: `temporal/activities/`
- Shared strategy: One file in `activities/`
- Per-service strategy: Multiple files in `activities/`

**Service Implementation:**
- Workflow goes in: `temporal/workflows/`
- Activities go in: `temporal/activities/`
- Worker goes in: `temporal/workers/`

**Worker Configuration:**
- Always in: `temporal/workers/`
- Name: `main_worker.[ext]` or `[service]_worker.[ext]`

**Tests:**
- Workflow tests: `temporal/tests/workflows/`
- Activity tests: `temporal/tests/activities/`
- Integration tests: `temporal/tests/integration/`

### README Generation

**ALWAYS generate `temporal/README.md` with:**
- Folder structure explanation
- Implementation type (API or Service)
- Target name (which API or service)
- Language used
- Getting started instructions
- Configuration requirements
- Testing instructions

**Example README:**
```markdown
# Temporal Implementation

This folder contains Temporal workflows and activities for [Target Name].

## Folder Structure

[Include folder tree]

## Implementation Type

**Type:** API / SERVICE
**Target:** [Name]
**Language:** [Language]

## Getting Started

1. Install Temporal SDK
2. Configure .env
3. Run worker
4. Start workflows

## Configuration

See .env for required variables

## Testing

Run tests with [command]
```

### Organization Benefits

**Benefits of temporal/ folder:**
- ✅ **Clean separation** - Temporal code separate from application code
- ✅ **Easy discovery** - All Temporal files in one place
- ✅ **Clear structure** - Standard layout across projects
- ✅ **No confusion** - Never wonder where files go
- ✅ **Professional** - Industry-standard organization
- ✅ **Easy maintenance** - Update/refactor with confidence
- ✅ **Better testing** - Tests mirror implementation structure

---

## Code Generation Standards

### Workflow Code

**Requirements:**
- Clear workflow function/method naming (`processOrder`, `handlePayment`)
- Proper signal and query handlers
- Timeout configurations
- Versioning for safe deployments
- Deterministic code only
- No direct external calls (use activities)

**Example Structure:**
```typescript
import { proxyActivities } from '@temporalio/workflow';
import type * as activities from './activities';

const { activity1, activity2 } = proxyActivities<typeof activities>({
  startToCloseTimeout: '1 minute',
  retry: {
    maximumAttempts: 3,
  },
});

export async function myWorkflow(input: WorkflowInput): Promise<WorkflowResult> {
  // Step 1: Execute activity
  const result1 = await activity1(input.param1);
  
  // Step 2: Execute another activity
  const result2 = await activity2(result1);
  
  return { success: true, data: result2 };
}
```

### Activity Code

**Requirements:**
- Idempotent operations
- Appropriate timeout settings
- Retry policies with exponential backoff
- Proper error types for retry vs. non-retry failures
- External API calls and side effects
- Database operations

**Example Structure:**
```python
from temporalio import activity
from typing import Dict

@activity.defn
async def process_payment(order_id: str) -> Dict[str, any]:
    """
    Process payment for order.
    This activity is idempotent - safe to retry.
    """
    try:
        # External API call
        result = await payment_gateway.charge(order_id)
        return {"success": True, "transaction_id": result.id}
    except PaymentDeclinedError as e:
        # Non-retryable error
        raise ApplicationError(
            f"Payment declined: {e}",
            non_retryable=True
        )
    except PaymentGatewayError as e:
        # Retryable error (Temporal will retry automatically)
        raise ApplicationError(f"Payment gateway error: {e}")
```

### Worker Code

**Requirements:**
- Task queue configuration
- Graceful shutdown handling
- Resource management
- Health check endpoints
- Environment variable usage

**Example Structure:**
```go
package main

import (
    "log"
    "os"
    
    "go.temporal.io/sdk/client"
    "go.temporal.io/sdk/worker"
)

func main() {
    // Connect to Temporal
    c, err := client.Dial(client.Options{
        HostPort:  os.Getenv("TEMPORAL_URI"),
        Namespace: os.Getenv("TEMPORAL_NAMESPACE"),
    })
    if err != nil {
        log.Fatalln("Unable to create client", err)
    }
    defer c.Close()

    // Create worker
    w := worker.New(c, os.Getenv("TASK_QUEUE"), worker.Options{})

    // Register workflows and activities
    w.RegisterWorkflow(MyWorkflow)
    w.RegisterActivity(MyActivity)

    // Start worker with graceful shutdown
    err = w.Run(worker.InterruptCh())
    if err != nil {
        log.Fatalln("Unable to start worker", err)
    }
}
```

### Testing Code

**Requirements:**
- Unit tests for activities (mock external dependencies)
- Integration tests for workflows
- Test environment setup
- Time-skipping for long-running workflow tests

**Example Structure:**
```java
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.testing.TestWorkflowExtension;
import io.temporal.worker.Worker;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

public class MyWorkflowTest {
    @RegisterExtension
    public static final TestWorkflowExtension testWorkflow =
        TestWorkflowExtension.newBuilder()
            .setWorkflowTypes(MyWorkflowImpl.class)
            .setActivityImplementations(new MyActivitiesImpl())
            .build();

    @Test
    public void testWorkflow() {
        MyWorkflow workflow = testWorkflow.newWorkflowStub(MyWorkflow.class);
        WorkflowResult result = workflow.processOrder("order-123");
        assertEquals(true, result.isSuccess());
    }
}
```

---

## Configuration Best Practices

### Environment-Based Configuration

Always use environment variables loaded from `.env` file:

```bash
# Required: Connection details
TEMPORAL_URI=<server-address>:7233
TEMPORAL_NAMESPACE=<stage1|stage2|production>
TEMPORAL_API_KEY=<your-api-key>

# Required: Task queue
TASK_QUEUE=default

# Optional: For production with TLS
TEMPORAL_TLS_CERT_PATH=<path-to-cert>
TEMPORAL_TLS_KEY_PATH=<path-to-key>
```

### Development Environment

```bash
TEMPORAL_URI=localhost:7233
TEMPORAL_NAMESPACE=stage1
TEMPORAL_API_KEY=dev-api-key
TASK_QUEUE=dev-queue
```

### Production Environment

```bash
TEMPORAL_URI=temporal-prod.company.com:7233
TEMPORAL_NAMESPACE=production
TEMPORAL_API_KEY=prod-api-key-secure
TASK_QUEUE=prod-queue
TEMPORAL_TLS_CERT_PATH=/etc/temporal/certs/client.pem
TEMPORAL_TLS_KEY_PATH=/etc/temporal/certs/client-key.pem
```

---

## API vs Service Implementation Strategies

### API-Based Implementation

**When to Use:**
- External API is used across multiple services
- Want unified retry/timeout policies for all API calls
- Need centralized API observability
- API calls are scattered across codebase

**Implementation Approaches:**

**Option A: Shared Activity (Recommended for most cases)**
- Create ONE reusable activity for all calls to the API
- Example: `openai_api_activity.py` handles ALL OpenAI calls
- Benefits:
  - ✅ Centralized configuration (one retry policy)
  - ✅ Consistent error handling
  - ✅ Easier to update/maintain
  - ✅ Unified observability
- Use when: API behavior is consistent across services

**Option B: Per-Service Activities**
- Create separate activity per service using the API
- Example: `llm_service_openai_activity.py`, `chat_service_openai_activity.py`
- Benefits:
  - ✅ Service-specific behavior
  - ✅ Different retry policies per service
  - ✅ Service isolation
  - ✅ Easier to customize per service
- Use when: API behavior differs by service

**Interactive Selection:**
When developer approves API implementation, ASK:
```
You selected: [API Name] (X calls across Y files)

Implementation Strategy:
A) Shared Activity - One reusable activity for all services
B) Per-Service Activities - Separate activity per service

Which approach? [A/B]: _
```

**Implementation Example (Shared Activity):**

```python
# temporal/activities/openai_api_activity.py
from temporalio import activity
from typing import Dict, Any
import openai

@activity.defn
async def call_openai_api(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shared OpenAI API activity - handles all OpenAI calls
    """
    activity.logger.info(f"OpenAI API call: {request}")
    
    try:
        response = await openai.ChatCompletion.acreate(
            model=request['model'],
            messages=request['messages']
        )
        return {'success': True, 'response': response}
    except openai.error.RateLimitError as e:
        activity.logger.warning(f"Rate limit hit: {e}")
        raise  # Temporal will retry
    except openai.error.APIError as e:
        activity.logger.error(f"API error: {e}")
        raise
```

### Service-Based Implementation

**When to Use:**
- Want to convert entire service to Temporal
- Service has multiple functions that work together
- Business logic is cohesive within one file/module
- Want complete workflow orchestration

**Implementation Approach:**
- Create workflow that orchestrates service operations
- Create activities for each service function
- Example: Payment Service → Payment Workflow + Activities

**Implementation Example:**

```python
# temporal/workflows/payment_service_workflow.py
from temporalio import workflow
from datetime import timedelta

@workflow.defn
class PaymentServiceWorkflow:
    @workflow.run
    async def run(self, payment_data: Dict) -> Dict:
        # Orchestrate payment service operations
        validated = await workflow.execute_activity(
            validate_payment,
            payment_data,
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        charged = await workflow.execute_activity(
            charge_payment,
            validated,
            start_to_close_timeout=timedelta(seconds=30)
        )
        
        return charged

# temporal/activities/payment_service_activities.py
@activity.defn
async def validate_payment(data: Dict) -> Dict:
    # Validation logic
    pass

@activity.defn
async def charge_payment(data: Dict) -> Dict:
    # Charging logic
    pass
```

### Mixed Implementation (Service + API)

**When to Use:**
- Service uses external APIs
- Want service orchestration + API resilience

**Implementation Approach:**
- Create service workflow
- Create service activities
- Use shared API activities within service activities

**Example:**

```python
# temporal/workflows/llm_service_workflow.py
@workflow.defn
class LLMServiceWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> Dict:
        # Orchestrate LLM service
        preprocessed = await workflow.execute_activity(
            preprocess_prompt,
            prompt,
            start_to_close_timeout=timedelta(seconds=5)
        )
        
        # Use shared OpenAI activity
        result = await workflow.execute_activity(
            call_openai_api,
            {'model': 'gpt-4', 'messages': [preprocessed]},
            start_to_close_timeout=timedelta(seconds=60)
        )
        
        return result
```

### Selection Guidelines

**Choose API Implementation When:**
- API calls are scattered across multiple files
- Need unified API configuration
- Want centralized API monitoring
- API is used by 3+ services

**Choose Service Implementation When:**
- Service has cohesive business logic
- Multiple functions work together
- Need workflow orchestration
- Service has 3+ functions to convert

**Choose Mixed When:**
- Service uses external APIs
- Want both orchestration and API resilience
- Complex business logic + external dependencies

---

## Language-Specific Guidelines

```### Dynamic Language Detection

- Detect project's primary language from file extensions, package files
- Use language-appropriate Temporal SDK
- Follow language-specific best practices and idioms

### Package/Dependency Management (January 2026)

**TypeScript:** `package.json` with `@temporalio/client`, `@temporalio/worker`, `@temporalio/workflow` v1.10.0+
**Python:** `requirements.txt` with `temporalio` v1.5.0+
**Go:** `go.mod` with `go.temporal.io/sdk` v1.25.1+
**Java:** `pom.xml` with `io.temporal:temporal-sdk` v1.23.0+
**.NET:** `.csproj` with `Temporalio` v1.1.0+
**PHP:** `composer.json` with `temporal/sdk` v2.10.0+
**Ruby:** `Gemfile` with `temporalio` v0.1.0+ (Preview)

---

## Security Considerations

**Always Include:**
- Environment variables for credentials (never hardcode)
- Data encryption for sensitive workflow data
- mTLS for production deployments
- Principle of least privilege for worker permissions
- Input validation in activities
- Secure secret management

---

## Observability & Monitoring

**Always Include:**
- Structured logging in activities
- Metric collection (success/failure rates, latency)
- Workflow execution tracing
- Alert configuration examples
- Health check endpoints
- Status query handlers

---

## Error Handling Philosophy

**Transient Errors:** Let Temporal retry automatically
```python
# Retryable - Temporal handles this
raise ApplicationError("Database connection timeout")
```

**Permanent Errors:** Fail fast with clear error messages
```python
# Non-retryable - Don't waste time retrying
raise ApplicationError("Invalid email format", non_retryable=True)
```

**Compensation:** Implement undo/rollback activities
```typescript
try {
  await chargePayment(orderId);
  await reserveInventory(orderId);
} catch (error) {
  await refundPayment(orderId);  // Compensate
  throw error;
}
```

**Timeouts:** Set realistic timeouts at every level
```go
opts := workflow.ActivityOptions{
    StartToCloseTimeout: 30 * time.Second,
    RetryPolicy: &temporal.RetryPolicy{
        MaximumAttempts: 3,
    },
}
```

---

## Documentation Standards

**When implementing Temporal code, always include:**

### Inline Comments
- Explain workflow logic
- Document retry behavior
- Note idempotency requirements
- Describe compensation logic

### README File
Include sections for:
- **Setup Instructions** - How to configure environment
- **Running Workers** - Commands to start workers
- **Starting Workflows** - How to trigger workflows
- **Monitoring** - How to view execution status
- **Troubleshooting** - Common issues and solutions

---

## Anti-Patterns to Avoid

### ❌ Don't Use Workflows For:

**Simple synchronous operations**
```typescript
// BAD: Too simple for Temporal
function addNumbers(a: number, b: number): number {
  return a + b;
}
```

**Pure data transformations**
```python
# BAD: No external interactions
def uppercase_strings(items: list) -> list:
    return [item.upper() for item in items]
```

**High-frequency operations (>100/sec)**
```go
// BAD: Too high frequency
func logMetric(metric Metric) {
    // Called 1000s of times per second
}
```

**Operations completing in <1 second**
```java
// BAD: Too fast, adds unnecessary overhead
public String formatDate(Date date) {
    return dateFormatter.format(date);
}
```

### ❌ Don't Do in Workflow Code:

**Direct external API calls** (use activities instead)
```typescript
// BAD: External call in workflow
export async function myWorkflow() {
  const result = await fetch('https://api.example.com');  // ❌ WRONG
}

// GOOD: External call in activity
export async function myWorkflow() {
  const result = await proxyActivity.callExternalAPI();  // ✅ CORRECT
}
```

**Non-deterministic code**
```python
# BAD: Non-deterministic
async def my_workflow():
    random_value = random.random()  # ❌ WRONG
    current_time = datetime.now()   # ❌ WRONG
    uuid = uuid.uuid4()             # ❌ WRONG
```

**Threading/concurrency primitives**
```go
// BAD: Manual threading
func MyWorkflow(ctx workflow.Context) error {
    go func() {  // ❌ WRONG - use workflow.Go instead
        // background task
    }()
}
```

---

## Automatic Response Format

**When implementing Temporal (automatically, without prompting):**

1. **Silent Detection** - Identify pattern internally
2. **Registry Check** - Verify no duplicate exists
3. **Configuration Check** - Validate setup
4. **Immediate Implementation** - Generate complete code
5. **Full Solution** - Include workflows, activities, workers, client, config, tests
6. **Environment Integration** - Use `.env` variables automatically
7. **Best Practices** - Apply error handling, retry policies, logging
8. **Create Backup** - Backup generated files before cleanup
9. **Run Cleanup** - Remove emojis/images from codebase
10. **Validate Cleanup** - Check cleaned files are valid
11. **Update Registry** - Add entry with all details (only if cleanup succeeds)
12. **Update State** - Modify `summary.md` to reflect changes

**Developer Experience:** Code appears complete and ready to use without any back-and-forth

---

## Continuous Learning

- Stay updated with Temporal best practices
- Reference official Temporal documentation patterns
- Adapt suggestions based on developer feedback
- Prioritize developer experience and code clarity

---

**Last Updated:** January 14, 2026 (Added temporal/ folder structure, API vs Service implementation strategies, naming conventions, file organization rules)

