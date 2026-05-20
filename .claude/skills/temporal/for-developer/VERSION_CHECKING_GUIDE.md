# SDK Version Checking & Alerts Guide

## How Version Checking Works

### Dynamic Version Detection (Not Hardcoded)

Cursor AI automatically checks SDK versions dynamically:

1. **Reads Minimum Versions from Documentation**
   - Reads from `cursor-skills-temporal/for-cursor/temporal-overview.mdc` (SDK Language Support section)
   - Fallback: Reads from `cursor-skills-temporal/for-cursor/summary.md` (SDK Versions section)
   - **Never hardcodes versions** - always reads from documentation files

2. **Detects Installed Versions**
   - Reads from `package.json` (TypeScript/JavaScript)
   - Reads from `requirements.txt` or `pyproject.toml` (Python)
   - Reads from `go.mod` (Go)
   - Reads from `pom.xml` or `build.gradle` (Java)
   - Reads from `.csproj` (.NET)
   - Reads from `composer.json` (PHP)
   - Reads from `Gemfile` (Ruby)

3. **Compares Versions**
   - Compares installed version vs minimum required version
   - If outdated: Adds to validation feedback
   - If missing: Adds to validation feedback

### Current Minimum SDK Versions (Auto-Fetched from https://docs.temporal.io/)

Versions are automatically fetched from Temporal documentation. Example minimums:

- **TypeScript:** 1.10.0+ (Updates API, Nexus support)
- **Python:** 1.5.0+ (enhanced async, type hints)
- **Go:** 1.25.0+ (Nexus support, Updates API)
- **Java:** 1.23.0+ (Spring integration, Updates API)
- **.NET:** 1.1.0+ (async improvements)
- **PHP:** 2.10.0+ (PHP 8.2+ support)
- **Ruby:** 0.1.0+ (Preview/Beta)

**Note:** Cursor automatically fetches latest versions from https://docs.temporal.io/ using web_search tool. No manual updates needed - system stays current automatically.

## Where Alerts Appear

### Alert Location: Cursor Chat Window

When Cursor detects a Temporal pattern but finds configuration issues, alerts appear **directly in the Cursor chat window** as a message.

### Alert Trigger

Alerts appear when:
1. Developer writes code that would benefit from Temporal
2. Cursor detects the pattern automatically
3. Cursor validates configuration (runs automatically)
4. Issues found: Missing dependencies, outdated versions, invalid config
5. Alert message appears in chat with specific fixes

### Alert Format

```
⚠️ TEMPORAL CONFIGURATION INCOMPLETE

I detected code that would benefit from Temporal workflows, but found these issues:

Configuration Issues:
❌ TEMPORAL_URI: Currently "localhost:7233" 
   → Change to: your-company-temporal-server.com:7233
   → Reason: Stage2/Production requires company server

Dependency Issues:
❌ Missing: @temporalio/client
   → Run: npm install @temporalio/client@^1.10.0 @temporalio/worker@^1.10.0 @temporalio/workflow@^1.10.0 dotenv
   → Reason: TypeScript project requires Temporal SDK packages

❌ Outdated: @temporalio/client (installed: 1.8.0, required: 1.10.0+)
   → Run: npm install @temporalio/client@^1.10.0 @temporalio/worker@^1.10.0 @temporalio/workflow@^1.10.0
   → Reason: Current version missing Updates API and Nexus support (latest features)
   → Impact: Some features may not work correctly

Required Actions (in order):
1. Edit .env file and set: [list each missing variable]
2. Install/update dependencies: [show exact command for detected language with versions]
3. Verify connection: [provide validation command]

After completing these steps, I will automatically implement your Temporal workflow.
```

### What Gets Checked

1. **Environment Variables** (.env file)
   - TEMPORAL_URI
   - TEMPORAL_NAMESPACE
   - TLS certificates (if production)
   - Other namespace-specific variables

2. **Dependencies** (package files)
   - Missing packages
   - Outdated SDK versions
   - Version compatibility

3. **Configuration Values**
   - Valid namespace values
   - Server addresses (not localhost for production)
   - TLS requirements for production

### Alert Behavior

- **Automatic:** No developer prompt needed - Cursor checks automatically
- **Dynamic:** Alerts adapt to detected language, namespace, project structure
- **Complete:** Lists ALL issues, not just the first one found
- **Actionable:** Provides specific commands for detected environment
- **Non-blocking:** If config is 100% valid, Cursor implements silently (no alert)

## Automatic Version Updates (Fully Dynamic)

### How It Works (Fully Automatic - No Manual Updates Needed)

The system automatically fetches latest SDK versions from Temporal's official sources:

1. **Automatic Fetching:**
   - Cursor automatically checks Temporal documentation and package registries
   - Fetches latest stable versions during validation
   - No manual updates needed - system stays current automatically

2. **Fallback to Local Docs:**
   - If online fetch fails, uses local documentation files
   - Reliable offline operation
   - Local docs updated as backup

3. **Version Caching:**
   - Caches fetched versions for offline use
   - Reduces repeated fetches
   - Updates cache when new versions detected

### Manual Updates (Optional - Only If Needed)

If you want to manually update local documentation (for offline use or backup):

1. **Update Documentation Files:**
   - Edit `cursor-skills-temporal/for-cursor/temporal-overview.mdc`
   - Update SDK Language Support section with new minimum versions
   - Or update `cursor-skills-temporal/for-cursor/summary.md` SDK Versions section

2. **Cursor Uses Updated Versions:**
   - Cursor reads versions from documentation as fallback
   - Used if online fetch unavailable

### Example: Automatic Update

When Temporal releases TypeScript SDK 1.12.0:

1. **Automatic (No Action Needed):**
   - Cursor fetches from npm registry
   - Detects 1.12.0 as latest version
   - Uses 1.12.0+ automatically in next validation
   - Alerts developers if they have older versions

2. **Manual Update (Optional - For Offline Backup):**
   - Edit `temporal-overview.mdc`:
   ```
   #### **TypeScript SDK**
   - **Package**: `@temporalio/client`, `@temporalio/worker`, `@temporalio/workflow` v1.12.0+ (MINIMUM REQUIRED)
   ```

## Version Checking Protocol

### For Cursor AI

1. **Read Minimum Versions:**
   - Read from `{DISCOVERED_PATH}/for-cursor/temporal-overview.mdc` (SDK Language Support section)
   - Fallback: `{DISCOVERED_PATH}/for-cursor/summary.md` (SDK Versions section)

2. **Detect Installed Versions:**
   - Parse package files based on detected language
   - Extract version numbers

3. **Compare:**
   - If installed < minimum: Add to validation feedback
   - If missing: Add to validation feedback
   - If check fails: Warn but continue

4. **Include in Alert:**
   - Show installed version
   - Show required version
   - Provide update command
   - Explain impact (missing features)

## Summary

- **Version Checking:** Fully automatic - fetches from Temporal sources (docs, package registries)
- **Alert Location:** Cursor chat window
- **Alert Trigger:** Automatic when pattern detected and config invalid
- **Alert Content:** Missing dependencies, outdated versions, invalid config
- **Update Process:** Fully automatic - no manual updates needed (system fetches latest versions)
- **Offline Support:** Falls back to local documentation if fetch fails

