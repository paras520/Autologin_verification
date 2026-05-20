# ENVIRONMENT CONFIGURATION GUIDE

## ⚠️ IMPORTANT: Pre-Filled Credentials Available

The system now includes **pre-filled shared dev/staging credentials** that work immediately!

When you run `quick-start.py`, it creates `.env` with working credentials for a shared Temporal server.

**You have two options:**
- **Option A:** Use pre-filled credentials (ready to use immediately)
- **Option B:** Customize with your own Temporal server

---

## Quick Setup (Pre-Filled Credentials)

### Step 1: Run Quick-Start Script

```bash
cd cursor-skills-temporal
python scripts/quick-start.py
```

This creates `.env` with:
```env
TEMPORAL_URI=temporal.diro.live:7233
TEMPORAL_NAMESPACE=stage1
TEMPORAL_API_KEY=7aa987332caca3271582d78d78259c7d3743fa1d954619c7f8236d2f04bd5978
TASK_QUEUE=temporal-task-queue
```

### Step 2: Choose Your Option

When you open Cursor, it will prompt:
```
📊 Current Temporal Configuration
TEMPORAL_URI: temporal.diro.live:7233
...

Would you like to:
A) Keep these settings (continue)
B) Change to your own Temporal server
```

**Choose A to use shared server (recommended for getting started)**

---

## Custom Setup (Your Own Server)

If you choose Option B or want to customize later:

### Step 1: Create the File

Create a new file named **`.env`** (with the dot) in your project root directory.

### Step 2: Copy This Template

Copy this entire block into your `.env` file:

```env
# =============================================================================
# REQUIRED: Temporal Server Connection
# =============================================================================

# Temporal server address (with port :7233)
# Development (stage1): Use localhost:7233 or your dev server
# Staging/Production: Use your company's Temporal server address
TEMPORAL_URI=localhost:7233

# Temporal namespace - determines which environment you're connecting to
# Options: stage1 (dev) | stage2 (staging) | production
TEMPORAL_NAMESPACE=stage1

# Task queue name - default is usually sufficient
TASK_QUEUE=default

# =============================================================================
# OPTIONAL: TLS Configuration (Required for Production)
# =============================================================================

# Path to client certificate (required for production namespace)
TEMPORAL_TLS_CERT_PATH=

# Path to client private key (required for production namespace)
TEMPORAL_TLS_KEY_PATH=

# Path to CA certificate (optional, for custom certificate authorities)
TEMPORAL_TLS_CA_PATH=

# =============================================================================
# OPTIONAL: Worker Configuration
# =============================================================================

# Maximum concurrent activity executions per worker
MAX_CONCURRENT_ACTIVITIES=100

# Maximum concurrent workflow executions per worker
MAX_CONCURRENT_WORKFLOWS=50

# =============================================================================
# OPTIONAL: Application Configuration
# =============================================================================

# Logging level: debug | info | warn | error
LOG_LEVEL=info

# Application environment
NODE_ENV=development
```

### Step 3: Update Values

Replace the placeholder values with your actual Temporal server details.

---

## Configuration Examples by Environment

### Development (stage1)

```env
TEMPORAL_URI=localhost:7233
TEMPORAL_NAMESPACE=stage1
TASK_QUEUE=default
# No TLS required for local development
```

OR if you have a dev server:

```env
TEMPORAL_URI=temporal-dev.your-company.com:7233
TEMPORAL_NAMESPACE=stage1
TASK_QUEUE=default
```

### Staging (stage2)

```env
TEMPORAL_URI=temporal-staging.your-company.com:7233
TEMPORAL_NAMESPACE=stage2
TASK_QUEUE=default
# TLS optional - check with your team
```

### Production

```env
TEMPORAL_URI=temporal-prod.your-company.com:7233
TEMPORAL_NAMESPACE=production
TASK_QUEUE=default

# TLS REQUIRED for production
TEMPORAL_TLS_CERT_PATH=/etc/temporal/certs/client.pem
TEMPORAL_TLS_KEY_PATH=/etc/temporal/certs/client-key.pem
TEMPORAL_TLS_CA_PATH=/etc/temporal/certs/ca.pem
```

---

## What Each Variable Means

### Required Variables

**TEMPORAL_URI**
- The address of your Temporal server with port
- Format: `hostname:7233` (port is almost always 7233)
- Examples:
  - Local: `localhost:7233`
  - Remote: `temporal.company.com:7233`

**TEMPORAL_NAMESPACE**
- Which Temporal namespace (environment) to connect to
- Valid values: `stage1`, `stage2`, `production`
- This determines which environment your workflows run in

**TASK_QUEUE**
- The name of the task queue your workers will poll
- Default: `default` (works for most cases)
- Use custom names to route work to specialized workers

### Optional Variables (Production)

**TEMPORAL_TLS_CERT_PATH**
- Path to your TLS client certificate file
- Required for production namespaces
- Get from your infrastructure/DevOps team

**TEMPORAL_TLS_KEY_PATH**
- Path to your TLS private key file
- Required for production namespaces
- Get from your infrastructure/DevOps team

**TEMPORAL_TLS_CA_PATH**
- Path to the Certificate Authority certificate
- Optional, needed for custom CAs
- Get from your infrastructure/DevOps team

---

## Security Notes

### DO:
- Create `.env` file manually in your project root
- Fill it with your actual Temporal server credentials
- Keep it local - never commit to git
- Use different values for dev/staging/production
- Protect certificate files (chmod 600 on Linux/Mac)

### ❌ DON'T:
- Commit `.env` to version control (it's in .gitignore)
- Share certificate files publicly
- Use production credentials in development
- Hardcode credentials in source code

---

## Verification

After creating your `.env` file:

1. **Check File Exists:**
   - File is named exactly `.env` (with the dot, no extension)
   - Located in project root directory
   - Contains the required variables

2. **Cursor AI Will Validate:**
   - When you write code, Cursor will automatically check configuration
   - If something is missing, you'll get specific instructions
   - If everything is valid, implementation happens automatically

3. **Test Connection (Optional):**

   **TypeScript:**
   ```bash
   node -e "require('dotenv').config(); console.log(process.env.TEMPORAL_URI)"
   ```

   **Python:**
   ```bash
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('TEMPORAL_URI'))"
   ```

   **Go:**
   ```bash
   go run -e 'package main; import ("fmt"; "os"; "github.com/joho/godotenv"); func main() { godotenv.Load(); fmt.Println(os.Getenv("TEMPORAL_URI")) }'
   ```

---

## Troubleshooting

**Problem: Cursor says .env is missing**
- Solution: Create the file in the project root (same directory as README.md)
- Check filename is exactly `.env` (not `env.txt` or `.env.txt`)

**Problem: Connection refused**
- Solution: Verify `TEMPORAL_URI` is correct
- Check you have network access to the Temporal server (VPN?)
- For localhost, ensure Temporal is running

**Problem: Wrong namespace error**
- Solution: Verify `TEMPORAL_NAMESPACE` matches your intended environment
- Ask your team which namespace to use

**Problem: TLS certificate errors**
- Solution: Check certificate file paths are correct
- Verify files exist and are readable
- Ensure certificate files are valid (not expired)

---

## Need Help?

**Ask Your Team:**
- What is the Temporal server address (TEMPORAL_URI)?
- Which namespace should I use (TEMPORAL_NAMESPACE)?
- Do I need TLS certificates? Where do I get them?

**Check These Files:**
- `README.md` - Full system documentation
- `summary.md` - Current system status
- `SYSTEM_GUIDE.md` - How Cursor AI operates

---

## Example Commands to Create .env File

### Windows (PowerShell):
```powershell
# Navigate to project directory
cd "C:\path\to\cursor skills temporal"

# Create .env file
New-Item .env -ItemType File

# Edit with notepad
notepad .env
```

### Windows (Command Prompt):
```cmd
cd C:\path\to\cursor skills temporal
type nul > .env
notepad .env
```

### Mac/Linux (Terminal):
```bash
cd /path/to/cursor-skills-temporal
touch .env
nano .env
# or: vim .env
# or: code .env (VS Code)
```

Then paste the template from Step 2 above.

---

**After creating and configuring `.env`, you're ready! Cursor AI will automatically detect it and use your configuration.**

