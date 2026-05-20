# System Maintenance Guide

## Periodic Tasks (Over Time)

This guide lists all maintenance tasks needed to keep the Temporal Integration System up-to-date.

---

## 1. SDK Version Updates (Fully Automatic - No Manual Updates Needed)

### Automatic Version Fetching

**The system automatically fetches latest SDK versions from Temporal's official sources:**
- Temporal documentation (https://docs.temporal.io/)
- Package registries (npm, PyPI, Maven, Go, NuGet, etc.)
- GitHub releases

**No manual updates needed** - Cursor automatically:
1. Fetches latest versions during validation
2. Compares with installed versions
3. Alerts if outdated
4. Uses latest versions automatically

### Manual Updates (Optional - For Offline Backup Only)

If you want to update local documentation files for offline use:

**Step 1: Update Documentation Files**

Edit `cursor-skills-temporal/for-cursor/temporal-overview.mdc`:

Find the "SDK Language Support" section and update minimum versions:

```markdown
#### **TypeScript SDK**
- **Package**: `@temporalio/client`, `@temporalio/worker`, `@temporalio/workflow` v1.XX.0+ (MINIMUM REQUIRED)
```

**Step 2: Update Summary File (Optional)**

Edit `cursor-skills-temporal/for-cursor/summary.md`:

Find the "SDK Versions" section and update:

```markdown
- **TypeScript:** 1.XX.0+ (with Updates API, Nexus support)
```

**Note:** These updates are optional - system works automatically with online fetching.


### Example: Automatic Update

When Temporal releases TypeScript SDK 1.12.0:

1. **Automatic (No Action Needed):**
   - Cursor fetches from npm registry automatically
   - Detects 1.12.0 as latest version
   - Uses 1.12.0+ automatically in next validation
   - Alerts developers if they have older versions

2. **Manual Update (Optional - For Offline Backup):**
   - Edit `temporal-overview.mdc` and `summary.md` (only if you want offline backup)

---

## 2. Temporal Feature Updates (New APIs, Patterns)

### When to Update
- When Temporal releases new APIs (e.g., Updates API, Nexus Operations)
- When new workflow patterns become available
- When best practices change

### How to Update

**Update Knowledge Base Files:**

1. **`temporal-overview.mdc`**
   - Add new features to "Key Features & Capabilities" section
   - Update SDK versions if new features require minimum versions

2. **`temporal-workflows.mdc`**
   - Add new workflow patterns
   - Update examples with new APIs

3. **`temporal-activities.mdc`**
   - Add new activity patterns
   - Update retry policies if changed

4. **`temporal-language-examples.mdc`**
   - Add examples using new features
   - Update code examples for all languages

5. **`temporal-use-cases.mdc`**
   - Add new use case examples
   - Update existing use cases with new patterns

---

## 3. Cleanup Script Maintenance (NEW)

### Cleanup Script: `scripts/cleanup-code.py`

**Purpose:** Automatically removes emojis and images from codebase after Temporal implementation.

**Maintenance Tasks:**

1. **Backup Folder Cleanup:**
   - The cleanup script creates `.temporal-backups/` folder temporarily
   - This folder is automatically deleted after cleanup (success or failure)
   - If you see this folder lingering, it's safe to delete manually:
     ```bash
     rm -rf .temporal-backups/
     ```

2. **Script Updates:**
   - If cleanup patterns need updating (new emoji ranges, image formats)
   - Edit `scripts/cleanup-code.py`
   - Test with: `python scripts/cleanup-code.py [repo_root]`

3. **Validation:**
   - Cleanup script validates cleaned files automatically
   - If cleanup fails, original code is preserved
   - Registry is not updated if cleanup fails

**No regular maintenance needed** - Script runs automatically and handles errors gracefully.

---

## 4. Documentation Updates

### When to Update
- When Temporal documentation changes
- When new best practices emerge
- When security recommendations change

### Files to Review

- `temporal-deployment.mdc` - Production deployment patterns
- `temporal-error-handling.mdc` - Error handling strategies
- `temporal-testing.mdc` - Testing patterns
- `temporal-workers.mdc` - Worker configuration

---

## 4. Configuration Validation Updates

### When to Update
- When new environment variables are required
- When namespace requirements change
- When TLS requirements change

### How to Update

**Edit `CURSORRULES.txt`:**

Find "Configuration Validation Protocol" section and update:

```python
required_vars = {
    'NEW_VARIABLE': {
        'required': True,
        'validation': 'Description of what this variable does'
    }
}
```

---

## 5. Language Support Updates

### When to Update
- When new SDK languages are added
- When language-specific requirements change

### How to Update

1. **Add to `temporal-overview.mdc`** - SDK Language Support section
2. **Add to `temporal-language-examples.mdc`** - Complete working example
3. **Update `CURSORRULES.txt`** - Add language detection and dependency checking

---

## 6. System Health Checks

### Monthly Checks

1. **Verify File Structure**
   - Ensure all files exist in correct folders
   - Check file permissions

2. **Test Discovery**
   - Verify Cursor can discover `cursor-skills-temporal` folder
   - Test dynamic path resolution

3. **Review Registry**
   - Check `progress/temporal-registry.json` for errors
   - Verify tracking is working

### Quarterly Checks

1. **SDK Version Audit**
   - Check if Temporal released new SDK versions
   - Update minimum versions if needed

2. **Documentation Review**
   - Review Temporal official docs for changes
   - Update knowledge base files

3. **Validation Testing**
   - Test configuration validation with different languages
   - Verify alerts appear correctly

---

## 7. Troubleshooting Updates

### When Issues Are Found

1. **Document in `README.md`** - Add to troubleshooting section
2. **Update `DEVELOPER_SETUP_3_STEPS.md`** - Add common issues
3. **Update `CURSORRULES.txt`** - Fix validation logic if needed

---

## Maintenance Checklist

### Monthly
- [ ] Review Temporal release notes
- [ ] Check for SDK version updates
- [ ] Verify system health

### Quarterly
- [ ] Update SDK versions if needed
- [ ] Review and update documentation
- [ ] Test validation system
- [ ] Review registry for issues

### Annually
- [ ] Complete documentation audit
- [ ] Review all examples for accuracy
- [ ] Update all language examples
- [ ] Security review

---

## Notes

- **Version checking is dynamic** - Cursor reads from documentation files, not hardcoded values
- **No code changes needed** - Updating documentation files is sufficient
- **System is autonomous** - After initial setup, Cursor operates without developer input
- **Maintenance is minimal** - Most updates are documentation-only

---

## Quick Reference

**Most Common Update:**
1. Temporal releases new SDK version
2. Edit `temporal-overview.mdc` (SDK Language Support section)
3. Edit `summary.md` (SDK Versions section)
4. Done - Cursor uses new version automatically

**Files That Need Updates:**
- `for-cursor/temporal-overview.mdc` - SDK versions, features
- `for-cursor/summary.md` - SDK versions summary
- `for-cursor/temporal-*.mdc` - Feature updates, patterns
- `CURSORRULES.txt` - Validation logic, new requirements

**Files That Don't Need Updates:**
- `.cursorrules` (after initial setup)
- `progress/temporal-registry.json` (auto-updated by Cursor)
- `for-cursor/summary.md` (auto-updated by Cursor)
- Most developer documentation (unless major changes)

