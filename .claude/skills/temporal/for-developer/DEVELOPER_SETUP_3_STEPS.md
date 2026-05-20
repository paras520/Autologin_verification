# Developer Setup Guide - 3 Steps

---

## Quick Start

### Step 1: Drop Folder (10 seconds)
```bash
# Copy the cursor-skills-temporal folder into your code repository
cp -r cursor-skills-temporal /path/to/your-repo/

# Result:
your-repo/
├── src/                        ← Your code
├── cursor-skills-temporal/     ← Temporal system folder
```

---

### Step 2: Run Quick-Start Script (20 seconds) ⚡ RECOMMENDED!

**Automatic Setup (Easiest Method):**

```bash
cd /path/to/your-repo/cursor-skills-temporal
python scripts/quick-start.py
```

This script automatically:
- ✅ Copies `.cursorrules` from folder to repo root
- ✅ Copies `.gitignore` to repo root (or asks to merge)
- ✅ Creates `.env` template file
- ✅ Validates setup
- ✅ Shows next steps

**Or Manual Setup:**

```bash
cd /path/to/your-repo

# Copy .cursorrules from folder to repo root
cp cursor-skills-temporal/.cursorrules .cursorrules

# Copy .gitignore to repo root (or merge with existing)
cp cursor-skills-temporal/.gitignore .
```

**Result:**
```
your-repo/
├── .cursorrules                ← Cursor reads this automatically
├── .gitignore                  ← Security protection
├── .env                        ← Created by quick-start script
├── src/
└── cursor-skills-temporal/
    ├── .cursorrules            ← Source file (concise, ~150 lines)
    ├── for-cursor/
    │   └── references/         ← Detailed protocols
    └── scripts/
        └── quick-start.py      ← Setup script
```

**Why:** Cursor only reads `.cursorrules` from repository root!

---

### Step 3: Configure .env File (5 minutes)

**If you ran quick-start script, edit the template:**

```bash
# .env file was created automatically by quick-start.py
nano .env  # or your preferred editor
```

**If manual setup, create .env file:**

```bash
# Create .env file at repo root
nano .env  # or your preferred editor
```

**Add this content:**
```env
TEMPORAL_URI=your-temporal-server.com:7233
TEMPORAL_NAMESPACE=stage1
TASK_QUEUE=default
```

**Get these values from:**
- Your infrastructure/DevOps team
- Your Temporal server administrator
- See `cursor-skills-temporal/for-developer/ENV_SETUP_GUIDE.md` for details

**Save the file.**

---

## 🎯 What Happens Now?

After setup, Cursor will automatically:

1. **On Startup:**
   - Scan your entire codebase
   - Detect if Temporal is already implemented
   - Analyze existing Temporal efficiency
   - Show improvements in chat (if any)

2. **As You Code:**
   - Monitor new code for Temporal patterns
   - Automatically implement Temporal workflows
   - Clean code (remove emojis/images) after implementation
   - Update registry only if cleanup succeeds

3. **Always Active:**
   - Works in both new and existing codebases
   - Provides interactive approval for improvements
   - Maintains clean, production-ready code

---

## ✅ Validate Setup

Run the validation script to verify everything is correct:

```bash
cd cursor-skills-temporal
python scripts/validate-system.py
```

This checks:
- ✅ All required files present
- ✅ Folder structure correct
- ✅ .cursorrules file valid
- ✅ JSON files valid
- ✅ System ready for distribution

---

## Setup Complete

What happens now (Fully Automatic):
1. Open your project in Cursor IDE
2. Start a new chat session
3. Cursor automatically reads `.cursorrules` (built-in Cursor behavior)
4. Cursor automatically discovers `cursor-skills-temporal` folder (by fixed file names)
5. Cursor automatically reads `summary.md` (loads system state)
6. Cursor automatically reads `temporal-registry.json` (loads existing implementations)
7. Cursor automatically reads protocol reference files from `for-cursor/references/`
8. Cursor automatically loads Temporal knowledge from `.mdc` files
9. Write code normally
10. Cursor automatically detects patterns (no prompting needed)
11. Cursor automatically checks registry (avoids duplicates)
12. Cursor automatically validates config (dynamic check)
13. Cursor automatically implements Temporal workflows (if config valid)
14. Cursor automatically updates registry (tracks implementations)
15. Cursor automatically updates summary (maintains state)
16. Next chat session automatically continues from saved state

No manual Temporal coding required. Everything happens automatically.

---

## 📖 Need More Help?

**Quick Reference:**
- `cursor-skills-temporal/START_HERE.md` - Overview
- `cursor-skills-temporal/for-developer/QUICKSTART.md` - Detailed guide
- `cursor-skills-temporal/for-developer/README.md` - Complete documentation

**Troubleshooting:**
- Can't find `.cursorrules`? Make sure it's at repo root (same level as package.json)
- Cursor not detecting patterns? Restart Cursor IDE and start new chat
- Configuration errors? Check `cursor-skills-temporal/for-developer/ENV_SETUP_GUIDE.md`

---

## 🎯 What You Get

Automatic Temporal Integration:
- Long-running background jobs
- Multi-step business processes
- Distributed transactions (saga pattern)
- Microservice orchestration
- Scheduled/recurring tasks
- Error handling with retry logic

Write code - Cursor handles Temporal automatically.

