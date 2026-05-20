# Pattern Detection Rules

## Overview

This protocol defines how Cursor AI automatically detects code patterns suitable for Temporal implementation. Detection happens proactively without waiting for developer prompts.

---

## Detection Philosophy

**PROACTIVE:** Detect patterns before developer asks
**AUTOMATIC:** No prompts or confirmation needed
**INTELLIGENT:** Match patterns to appropriate Temporal services
**CONTEXT-AWARE:** Consider existing implementations (check registry)

---

## Automatic Detection Triggers

When you detect these patterns AND configuration is valid, IMMEDIATELY implement Temporal without waiting for developer confirmation:

---

## Two-Phase Detection Workflow

The detection process follows a two-phase approach for better performance and developer control:

**Phase 1: API Detection (Fast)**
- Scan for internal APIs (HTTP routes, GraphQL, gRPC)
- Scan for external APIs (imports, HTTP calls, config)
- Rank and prioritize APIs
- Present to developer for selection
- **Performance:** Fast - API-level scanning only

**Phase 2: Service Detection (Targeted)**
- Find services/functions using selected APIs from Phase 1
- Analyze usage patterns and frequency
- Generate implementation plan
- Present to developer for approval
- **Performance:** Targeted - only scans for selected APIs

**Benefits:**
- **Faster startup:** Phase 1 completes quickly
- **Developer control:** Choose which APIs to implement
- **Targeted analysis:** Phase 2 only analyzes relevant code
- **Clear separation:** APIs first, services second

---

## Pattern Categories

### Internal API Detection (Two-Phase Workflow - Phase 1)

When scanning codebase in Phase 1, detect internal APIs (routes/endpoints) for Temporal implementation:

**Internal API Types:**

1. **HTTP Routes (REST APIs):**
   - TypeScript/JavaScript: `app.post('/api/users', ...)`, `router.get('/api/orders/:id', ...)`
   - Python: `@app.route('/api/payment', methods=['POST'])`, `@app.post('/api/process')`
   - Go: `r.POST("/api/users", handler)`, `router.HandleFunc("/api/process", ...)`
   - Java: `@PostMapping("/api/users")`, `@GetMapping("/api/orders/{id}")`
   - C#: `[HttpPost("/api/payment")]`, `[Route("api/users")]`
   - PHP: `Route::post('/api/users', ...)`, `@Route("/api/payment")`
   - Ruby: `post '/api/users'`, `get '/api/orders/:id'`

2. **GraphQL Resolvers:**
   - TypeScript: `@Query()`, `@Mutation()`, `Query: { user: ... }`
   - Python: `@query`, `@mutation`, `def resolve_user(...)`
   - Any language: GraphQL schema definitions

3. **gRPC Methods:**
   - Go: `func (s *server) ProcessPayment(ctx context.Context, ...)`
   - Python: `def ProcessPayment(self, request, context):`
   - Java: `public void processPayment(..., StreamObserver<Response> observer)`

**Internal API Detection Method:**

- **Dynamic detection:** Scan for framework patterns across all languages
- **No hardcoded frameworks:** Detect Express, FastAPI, Flask, Spring, Rails, etc. automatically
- **Grouping:** Group by endpoint (deduplicateby path + method)
- **Priority:** Rank by occurrence count

### External API Detection (Two-Phase Workflow - Phase 1)

When scanning codebase in Phase 1, detect external APIs for targeted Temporal implementation:

**API Detection Methods:**

1. **Import/Library Detection:**
   - TypeScript/JavaScript: `import openai`, `import anthropic`, `require('stripe')`
   - Python: `import openai`, `from anthropic import Claude`, `import stripe`
   - Go: `"github.com/openai/openai-go"`, `"github.com/stripe/stripe-go"`
   - Java: `import com.openai.*`, `import com.stripe.*`
   - All languages: Check for popular API libraries

2. **HTTP Call Detection:**
   - Pattern: `fetch('https://api.openai.com/...')` 
   - Pattern: `requests.post('https://api.stripe.com/...')`
   - Pattern: `axios.get('https://api.example.com/...')`
   - Extract API name from URL domain

3. **Configuration Detection:**
   - `.env` files with API keys: `OPENAI_API_KEY`, `STRIPE_SECRET_KEY`
   - Config files with API endpoints
   - Environment variables for external services

**API Categories to Detect:**

**AI/LLM APIs:**
- OpenAI (GPT-3.5, GPT-4, Embeddings, DALL-E)
- Anthropic (Claude)
- Google (Gemini, PaLM)
- Cohere
- Llama/Replicate
- Stability AI

**Payment APIs:**
- Stripe
- PayPal
- Square
- Braintree

**Communication APIs:**
- Twilio (SMS, Voice)
- SendGrid (Email)
- Mailchimp
- Slack
- Discord

**Cloud Provider APIs:**
- AWS (S3, Lambda, SQS, etc.)
- Google Cloud Platform
- Azure
- DigitalOcean

**Database APIs:**
- MongoDB Atlas
- Supabase
- Firebase
- PlanetScale

**General HTTP APIs:**
- Any external HTTP calls
- REST APIs
- GraphQL endpoints

**API Detection Priority:**

Rank APIs by:
1. **Call count** (most used = highest priority)
2. **File count** (spread across many files = higher priority)
3. **Failure risk** (external APIs without retries = high priority)

**Temporal Implementation for APIs:**

For each detected API, suggest:
- **Shared Activity:** One reusable activity for all calls to that API
- **Per-Service Activity:** Separate activity per service using the API
- **Benefits:** Unified retry policy, timeout configuration, error handling, observability

**Example Detection Output:**

```
External APIs Detected:

1. OpenAI API (15 calls across 4 files)
   ├─ llmService.py: generateText() [line 45]
   ├─ llmService.py: embedText() [line 120]
   ├─ chatbot.py: processQuery() [line 78]
   └─ analyzer.py: analyzeContent() [line 23]
   Priority: HIGH (no retry policies detected)

2. Stripe API (8 calls across 2 files)
   ├─ payment.py: createCharge() [line 56]
   └─ subscription.py: createSubscription() [line 89]
   Priority: MEDIUM (some retry logic exists)
```

**These APIs should be detected during codebase scanning and offered as implementation targets alongside services.**

---

### NEW: Service Detection (For Complete Implementation)

When scanning existing codebase, group patterns by file to identify services:

**Service Detection:**
- Group patterns by file/module
- Identify business logic files (services, controllers, handlers)
- Count patterns per file (long-running, multi-step, external API, etc.)
- Suggest implementing entire service with Temporal

**Service Priority:**

Rank services by:
1. **Pattern diversity** (multiple pattern types = higher priority)
2. **Function count** (more functions = more benefit)
3. **Complexity** (multi-step + external API = high priority)

**Example Detection Output:**

```
Services Detected:

1. LLM Processing Service (services/llmService.py)
   - Pattern: Long-running + Multi-step + External API
   - Functions: 5 functions to convert
   - Priority: HIGH

2. Payment Processing Service (services/paymentService.py)
   - Pattern: Multi-step + External API
   - Functions: 3 functions to convert
   - Priority: HIGH
```

**These services should be detected during codebase scanning and offered as implementation targets alongside APIs.**

---

### NEW: Existing Code Patterns (For Refactoring)

When scanning existing codebase, detect these patterns for conversion to Temporal:

**Long-running Operations:**
- Functions with `setTimeout`, `setInterval`, `sleep`, `delay`, `wait`
- Operations that take > 1 second
- Functions with time-based delays

**Multi-step Processes:**
- Sequential function calls (3+ steps)
- Functions with multiple `await` statements
- Sequential API calls or database operations

**External API Calls:**
- HTTP requests (`fetch`, `axios`, `requests`, etc.)
- Functions making external service calls
- API orchestration code

**Background Jobs:**
- Queue processing code
- Worker pool implementations
- Async job handlers
- Background task processors

**Scheduled Tasks:**
- Cron expressions
- Scheduled functions
- Recurring tasks
- Timer-based operations

**These patterns should be detected during codebase scanning and suggested for refactoring to Temporal.**

---

### 1. Background Processing Patterns

**Indicators:**
- Long-running operations (>30 seconds)
- Async job processing
- Task queues or worker pools
- Scheduled/recurring tasks
- Batch processing operations
- Cron expressions or scheduling logic

**Code Examples:**
```javascript
// Long-running operations
async function processVideo(videoId) {
    await transcodeVideo(videoId);  // Takes 5+ minutes
    await generateThumbnails(videoId);
    await uploadToStorage(videoId);
}

// Scheduled tasks
setInterval(() => {
    cleanupExpiredSessions();
}, 3600000);  // Every hour

// Batch processing
async function processBatch(items) {
    for (const item of items) {
        await processItem(item);  // Could take hours
    }
}
```

**Temporal Solution:** Workflow with activities for each step, automatic retries, progress tracking

---

### 2. Reliability Requirements

**Indicators:**
- Manual retry logic
- Error recovery patterns
- Compensation logic (undo operations)
- Transactional workflows across services
- Need for guaranteed execution
- Durability requirements

**Code Examples:**
```python
# Manual retry logic
def call_external_api():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return requests.post(url, data=payload)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

# Compensation logic
async def processOrder(order_id):
    payment_captured = await capturePayment(order_id)
    try:
        await reserveInventory(order_id)
    except:
        await refundPayment(order_id)  # Compensate
        raise
```

**Temporal Solution:** Activity retry policies, automatic compensation workflows, saga pattern

---

### 3. Distributed System Patterns

**Indicators:**
- Microservice orchestration
- Saga patterns
- Multi-step business processes
- Service-to-service coordination
- Event-driven architectures
- Cross-service transactions

**Code Examples:**
```typescript
// Microservice orchestration
async function fulfillOrder(orderId: string) {
    await paymentService.charge(orderId);
    await inventoryService.reserve(orderId);
    await shippingService.createShipment(orderId);
    await notificationService.sendConfirmation(orderId);
}

// Saga pattern
async function bookTrip(tripId: string) {
    const flight = await bookFlight(tripId);
    try {
        const hotel = await bookHotel(tripId);
        try {
            const car = await bookCar(tripId);
        } catch {
            await cancelHotel(hotel.id);
            await cancelFlight(flight.id);
        }
    } catch {
        await cancelFlight(flight.id);
    }
}
```

**Temporal Solution:** Workflow orchestration, automatic saga pattern, compensation activities

---

### 4. State Management Needs

**Indicators:**
- Long-lived processes (hours/days/months)
- Durable execution requirements
- Process state persistence
- Workflow history tracking
- Resume after crash/restart
- Multi-day processes

**Code Examples:**
```go
// Long-lived process
func onboardNewEmployee(employeeID string) {
    createAccount(employeeID)
    // Wait 1 day for HR to complete paperwork
    time.Sleep(24 * time.Hour)
    setupWorkstation(employeeID)
    // Wait until training completed (could be weeks)
    waitForTrainingComplete(employeeID)
    grantFullAccess(employeeID)
}

// Process state persistence
type OrderState struct {
    OrderID    string
    Status     string
    LastUpdate time.Time
    RetryCount int
}
// Saving/loading state manually from database
```

**Temporal Solution:** Workflow with durable state, automatic persistence, built-in history tracking

---

### 5. Integration Scenarios

**Indicators:**
- External API calls with retry logic
- Database transactions across services
- File processing pipelines
- Email/notification sequences
- Payment processing flows
- Third-party service integration
- Webhook handling

**Code Examples:**
```java
// External API calls
public void processPayment(String orderId) {
    // Call payment gateway
    PaymentResult result = paymentGateway.charge(orderId);
    
    // Update multiple systems
    orderService.updateStatus(orderId, "paid");
    inventoryService.allocate(orderId);
    emailService.sendReceipt(orderId);
    analyticsService.trackConversion(orderId);
}

// File processing pipeline
public void processUploadedFile(String fileId) {
    File file = downloadFile(fileId);
    validateFile(file);
    parseData(file);
    transformData(file);
    loadToDatabase(file);
    sendNotification(fileId, "complete");
}
```

**Temporal Solution:** Activities for each integration, automatic retries, failure handling

---

## Use Case Matching

### Decision Tree

When pattern detected, reference `temporal-use-cases.mdc` to determine best implementation:

**Step 1: Categorize the Pattern**
- What type of pattern is it? (background, reliability, distributed, state, integration)
- What are the key requirements? (duration, retries, compensation, coordination)

**Step 2: Select Temporal Services**
- Workflow: Orchestration and state management
- Activities: External interactions and side effects
- Signals: External events and updates
- Queries: Status checks
- Schedules: Recurring executions

**Step 3: Check Registry**
- Search `temporal-registry.json` for similar implementations
- If duplicate exists: Reference existing instead of creating new
- If unique: Proceed with new implementation

**Step 4: Choose Pattern**
- Simple workflow: Single activity execution
- Multi-step workflow: Sequential activities
- Parallel execution: Concurrent activities
- Saga pattern: With compensation
- Long-running: With signals/queries
- Scheduled: With cron schedule

---

## Detection Examples

### Example 1: Order Processing

**Developer Code:**
```typescript
async function processOrder(orderId: string) {
    const order = await getOrder(orderId);
    await chargePayment(order);
    await updateInventory(order);
    await createShipment(order);
    await sendConfirmation(order);
}
```

**Detected Pattern:** Multi-step business process with external integrations
**Temporal Solution:** Workflow with 4 activities, retry policies, compensation logic

---

### Example 2: Report Generation

**Developer Code:**
```python
def generate_monthly_report():
    data = fetch_data_from_database()  # Takes 10 minutes
    report = process_data(data)  # Takes 30 minutes
    pdf = create_pdf(report)  # Takes 5 minutes
    send_email_with_attachment(pdf)
```

**Detected Pattern:** Long-running batch processing
**Temporal Solution:** Workflow with 4 activities, progress tracking, failure recovery

---

### Example 3: User Signup Flow

**Developer Code:**
```go
func handleUserSignup(email string) {
    createAccount(email)
    sendVerificationEmail(email)
    // Wait for email verification (could be days)
    waitForVerification(email)
    sendWelcomeEmail(email)
    createDefaultSettings(email)
}
```

**Detected Pattern:** Long-lived process with wait states
**Temporal Solution:** Workflow with signals for verification, durable wait states

---

## Anti-Patterns (Do Not Implement)

**Don't use Temporal for:**

❌ **Simple synchronous operations**
```typescript
function calculateTotal(items: Item[]): number {
    return items.reduce((sum, item) => sum + item.price, 0);
}
```

❌ **Pure data transformations**
```python
def transform_data(input_data):
    return [x.upper() for x in input_data]
```

❌ **High-frequency operations (>100/sec)**
```go
func handleMetric(metric Metric) {
    // Called thousands of times per second
    processMetric(metric)
}
```

❌ **Operations completing in <1 second**
```java
public String formatName(String firstName, String lastName) {
    return firstName + " " + lastName;
}
```

**Why:** Temporal adds overhead and complexity. These patterns are better served by direct function calls.

---

## Registry Check Protocol

**Before implementing ANY Temporal code:**

1. **Search Registry:** Check `temporal-registry.json` for similar implementations
   - Same workflow name?
   - Similar task queue?
   - Similar description?
   - Same activities?

2. **If Duplicate Found:**
   - Reference existing implementation
   - Suggest using existing workflow
   - Don't create new duplicate code

3. **If Unique:**
   - Proceed with new implementation
   - Update registry after completion

**Example Registry Check:**
```python
def check_registry_for_duplicate(workflow_name, task_queue):
    registry = load_registry()
    for workflow in registry['implementations']['workflows']:
        if workflow['name'] == workflow_name:
            return workflow  # Duplicate found
        if workflow['taskQueue'] == task_queue and similar_description(workflow):
            return workflow  # Similar workflow found
    return None  # Unique, proceed with implementation
```

---

## Proactive Suggestions

### When Developer Writes:

**Pattern:** `setTimeout`, `setInterval`, cron expressions
**Suggest:** Temporal scheduled workflows or cron workflows

**Pattern:** Manual retry logic with loops/delays
**Suggest:** Temporal activity retry policies

**Pattern:** Database state machines (status columns)
**Suggest:** Temporal workflows for state management

**Pattern:** Message queues (RabbitMQ, SQS) for job processing
**Suggest:** Temporal task queues and workflows

**Pattern:** Manual transaction coordination across services
**Suggest:** Temporal saga pattern implementation

**Pattern:** Webhook processing with retry logic
**Suggest:** Temporal activities with retry policies

---

## Implementation Style

**When Pattern Detected:**

1. **Silent Detection** - Identify pattern internally
2. **Registry Check** - Verify no duplicate exists
3. **Configuration Check** - Validate setup (see `validation-protocol.md`)
4. **Immediate Implementation** - Generate complete code if valid
5. **Full Solution** - Include workflows, activities, workers, client, config, tests
6. **Environment Integration** - Use `.env` variables automatically
7. **Best Practices** - Apply error handling, retry policies, logging
8. **Update Registry** - Add entry with all details
9. **Update State** - Modify `summary.md` to reflect changes

**Developer Experience:** Code appears complete and ready to use without any back-and-forth

---

## Key Principles

**PROACTIVE:** Implement before developer asks (if config valid)
**VALIDATED:** Check configuration before implementing
**COMPLETE:** Full workflows, activities, workers, tests, deployment
**AUTOMATIC:** Use `.env` config without asking
**SILENT:** No "should I implement this?" questions (unless config missing)
**FEEDBACK:** Alert developer if configuration incomplete
**NO DUPLICATES:** Always check registry first

---

**Last Updated:** January 14, 2026 (Added API detection patterns: External API detection methods, service grouping detection, API priority ranking, dual detection (services + APIs))

