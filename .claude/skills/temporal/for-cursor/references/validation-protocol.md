# Configuration Validation Protocol

## Overview

Before implementing ANY Temporal code, AUTOMATICALLY perform COMPLETE validation. This protocol defines dynamic, comprehensive configuration checking that adapts to the detected environment.

---

## Validation Philosophy

**DYNAMIC AND COMPREHENSIVE**
- Checks adapt based on namespace (stage1 vs production have different requirements)
- Checks adapt based on language (TypeScript needs different packages than Python)
- Version checking is dynamic - reads minimum versions from documentation, not hardcoded
- Version alerts included - outdated SDK versions appear in validation feedback
- Lists EVERY missing item, not just pre-defined ones
- Provides specific commands for detected environment

---

## Always Check Before Implementation

**Rule:** ONLY implement if validation passes 100%. If even ONE required item is missing, provide complete feedback instead.

---

## Step 1: Environment File Validation

### Check `.env` File

**Action:** Verify `.env` file exists at repository root and contains ALL required variables

### Required Variables

```python
# Core variables (always required)
required_vars = {
    'TEMPORAL_URI': {
        'required': True,
        'invalid_values': ['', 'localhost:7233'],  # For production/stage2
        'validation': 'Must be company Temporal server address with port :7233',
        'example': 'temporal-prod.company.com:7233'
    },
    'TEMPORAL_NAMESPACE': {
        'required': True,
        'valid_values': ['stage1', 'stage2', 'production'],
        'validation': 'Must be one of: stage1, stage2, production',
        'example': 'stage1'
    },
    'TEMPORAL_API_KEY': {
        'required': True,
        'validation': 'API key for authenticating with Temporal server',
        'example': 'your-api-key-here',
        'security_note': 'Keep this secret, never commit to version control'
    },
    'TASK_QUEUE': {
        'required': False,
        'default': 'default',
        'validation': 'Can use default value if not set',
        'example': 'my-task-queue'
    }
}
```

### Dynamic TLS Validation (Based on Namespace)

```python
# TLS requirements based on namespace
if namespace == 'production':
    required_vars['TEMPORAL_TLS_CERT_PATH'] = {
        'required': True,
        'validation': 'Production requires TLS certificate path',
        'example': '/etc/temporal/certs/client.pem'
    }
    required_vars['TEMPORAL_TLS_KEY_PATH'] = {
        'required': True,
        'validation': 'Production requires TLS private key path',
        'example': '/etc/temporal/certs/client-key.pem'
    }
```

---

## Step 2: Dynamic Dependency Validation

### Automatic Language Detection

**Action:** Detect project language from files, package managers, or project structure

**Detection Methods:**
- Check for `package.json` → TypeScript/JavaScript
- Check for `requirements.txt` or `pyproject.toml` → Python
- Check for `go.mod` → Go
- Check for `pom.xml` or `build.gradle` → Java
- Check for `.csproj` → .NET
- Check for `composer.json` → PHP
- Check for `Gemfile` → Ruby

### Language-Specific Requirements (with Version Checking)

```python
language_requirements = {
    'typescript': {
        'file': 'package.json',
        'packages': {
            '@temporalio/client': {
                'min_version': '1.10.0',
                'check_version': True
            },
            '@temporalio/worker': {
                'min_version': '1.10.0',
                'check_version': True
            },
            '@temporalio/workflow': {
                'min_version': '1.10.0',
                'check_version': True
            },
            'dotenv': {
                'min_version': None,
                'check_version': False
            }
        },
        'install_cmd': 'npm install @temporalio/client@^1.10.0 @temporalio/worker@^1.10.0 @temporalio/workflow@^1.10.0 dotenv'
    },
    'python': {
        'file': 'requirements.txt',
        'packages': {
            'temporalio': {
                'min_version': '1.5.0',
                'check_version': True
            },
            'python-dotenv': {
                'min_version': None,
                'check_version': False
            }
        },
        'install_cmd': 'pip install "temporalio>=1.5.0" python-dotenv'
    },
    'go': {
        'file': 'go.mod',
        'packages': {
            'go.temporal.io/sdk': {
                'min_version': 'v1.25.0',
                'check_version': True
            },
            'github.com/joho/godotenv': {
                'min_version': None,
                'check_version': False
            }
        },
        'install_cmd': 'go get go.temporal.io/sdk@v1.25.0 github.com/joho/godotenv'
    },
    'java': {
        'file': 'pom.xml',
        'packages': {
            'io.temporal:temporal-sdk': {
                'min_version': '1.23.0',
                'check_version': True
            }
        },
        'install_cmd': 'mvn dependency:resolve -Dartifact=io.temporal:temporal-sdk:1.23.0'
    }
    # Additional languages: .NET, PHP, Ruby...
}
```

---

## Step 3: SDK Version Validation (Auto-Fetch)

### Automatic Version Fetching

**Priority Order:**

1. **AUTOMATIC FETCH** - Use web_search tool to scan Temporal official documentation
   - Primary source: https://docs.temporal.io/
   - Search query: `site:docs.temporal.io SDK version latest [language]`
   - Parse version numbers from documentation pages automatically
   - SDK-specific pages:
     * TypeScript: https://docs.temporal.io/typescript
     * Python: https://docs.temporal.io/python
     * Go: https://docs.temporal.io/go
     * Java: https://docs.temporal.io/java
     * .NET: https://docs.temporal.io/dotnet
     * PHP: https://docs.temporal.io/php
     * Ruby: https://docs.temporal.io/ruby

2. **FALLBACK** - Package registries (if docs scan fails)
   - npm: https://www.npmjs.com/package/@temporalio/client
   - PyPI: https://pypi.org/project/temporalio/
   - Maven: https://mvnrepository.com/artifact/io.temporal/temporal-sdk
   - Go: https://pkg.go.dev/go.temporal.io/sdk
   - NuGet: https://www.nuget.org/packages/Temporalio

3. **FALLBACK** - Local documentation files
   - Read from `{DISCOVERED_PATH}/for-cursor/temporal-overview.mdc`
   - Reliable offline operation

4. **CACHE** - Store fetched versions for session
   - Reduces repeated fetches
   - Improves performance

5. **LAST RESORT** - Use documented minimums
   - TypeScript 1.10.0+, Python 1.5.0+, Go 1.25.0+, Java 1.23.0+, etc.
   - Only if all other sources fail

### Version Checking Logic

```python
def get_minimum_sdk_versions(detected_language):
    # Priority 1: Fetch from Temporal official source
    try:
        docs_result = web_search("site:docs.temporal.io SDK version latest " + detected_language)
        latest_versions = parse_versions_from_docs(docs_result, detected_language)
        if latest_versions:
            return latest_versions
    except:
        pass  # Fallback to next method
    
    # Priority 2: Fetch from package registry
    try:
        registry_versions = fetch_from_package_registry(detected_language)
        if registry_versions:
            return registry_versions
    except:
        pass  # Fallback to local docs
    
    # Priority 3: Read from local documentation
    min_versions = read_from_docs('{DISCOVERED_PATH}/for-cursor/temporal-overview.mdc')
    if min_versions:
        return min_versions
    
    # Priority 4: Use cached minimums
    return get_cached_minimums(detected_language)

# Check each installed package
for package in detected_temporal_packages:
    installed_version = get_version_from_package_file(package, detected_language)
    min_version = get_minimum_sdk_versions(detected_language)[package]
    
    if version_compare(installed_version, min_version) < 0:
        validation_results['outdated'].append({
            'package': package,
            'current': installed_version,
            'required': min_version,
            'update_cmd': generate_update_command(package, min_version, detected_language),
            'reason': f'SDK version outdated - need {min_version}+ for latest features'
        })
```

### Version Checking Protocol

**For each package:**
1. Read actual installed version from package file
2. Compare with minimum required version
3. If outdated: Add to validation feedback with update command
4. If missing: Add to validation feedback with install command
5. If version check fails: Warn but continue

**Critical:**
- Never hardcode versions - always fetch or read dynamically
- System automatically updates when Temporal releases new versions
- Works offline with cached/local docs fallback
- No manual documentation updates needed

---

## Step 4: Namespace-Specific Validation

### Different Rules per Namespace

```python
namespace_rules = {
    'stage1': {
        'allow_localhost': True,
        'tls_required': False,
        'description': 'Development environment',
        'uri_pattern': None  # Any URI allowed
    },
    'stage2': {
        'allow_localhost': False,
        'tls_required': False,
        'description': 'Staging environment - requires company server',
        'uri_pattern': r'^.+\.(company|internal)\.com:7233$'
    },
    'production': {
        'allow_localhost': False,
        'tls_required': True,
        'tls_ca_recommended': True,
        'description': 'Production - requires TLS and company server',
        'uri_pattern': r'^.+\.company\.com:7233$'
    }
}
```

---

## Step 5: Generate Dynamic Feedback

### If Configuration Incomplete or Invalid

**Create customized feedback based on what's actually missing or outdated:**

```
⚠️ TEMPORAL CONFIGURATION INCOMPLETE

I detected code that would benefit from Temporal workflows, but found these issues:

Configuration Issues:
[DYNAMICALLY LIST EVERY MISSING OR INVALID ITEM]

❌ TEMPORAL_URI: Currently "localhost:7233" 
   → Change to: your-company-temporal-server.com:7233
   → Reason: Stage2/Production requires company server

❌ TEMPORAL_NAMESPACE: Not set
   → Set to: stage1 (dev) | stage2 (staging) | production
   → Current phase determines server requirements

❌ TEMPORAL_TLS_CERT_PATH: Required for production
   → Set to: /path/to/client.pem
   → Get from: Infrastructure team

Dependency Issues:
❌ Missing: @temporalio/client
   → Run: npm install @temporalio/client@^1.10.0 @temporalio/worker@^1.10.0 @temporalio/workflow@^1.10.0 dotenv
   → Reason: TypeScript project requires Temporal SDK packages

❌ Outdated: @temporalio/client (installed: 1.8.0, required: 1.10.0+)
   → Run: npm install @temporalio/client@^1.10.0 @temporalio/worker@^1.10.0 @temporalio/workflow@^1.10.0
   → Reason: Current version missing Updates API and Nexus support (January 2026 features)
   → Impact: Some features may not work correctly

Required Actions (in order):
1. Edit .env file and set: [list each missing variable]
2. Install/update dependencies: [show exact command for detected language with versions]
3. Verify connection: [provide validation command]

After completing these steps, I will automatically implement your Temporal workflow.

Need help? Check {DISCOVERED_PATH}/for-developer/README.md for detailed setup instructions.
```

### Where Alerts Appear

**Alerts appear in Cursor chat when:**
- Configuration validation fails
- Pattern detected but config incomplete
- Includes ALL issues: missing dependencies, outdated versions, invalid config
- Provides specific commands for detected language and environment
- Developer sees alert → fixes issues → Cursor automatically implements

### If Configuration Valid

**If ALL validation checks pass:**
- Proceed with automatic implementation SILENTLY
- No alerts, no messages
- Just generate complete working code
- Update registry and summary automatically

---

## Validation Results Structure

```python
validation_results = {
    'valid': True/False,
    'errors': [],
    'warnings': [],
    'missing_vars': [],
    'invalid_vars': [],
    'missing_deps': [],
    'outdated_deps': [],
    'recommendations': []
}
```

---

## Key Principles

**DYNAMIC VALIDATION**
- Not a hardcoded checklist
- Adapts to detected language
- Adapts to detected namespace
- Checks EVERYTHING needed

**COMPREHENSIVE FEEDBACK**
- Lists EVERY problem found
- Provides specific solution for each
- Includes exact commands
- Shows expected vs. actual values

**VERSION AWARENESS**
- Auto-fetches latest minimum versions
- Compares installed vs. required
- Alerts on outdated packages
- Provides update commands

**INTELLIGENT CHECKING**
- Only checks what's relevant
- Stage1 doesn't need TLS
- Production requires TLS
- Language-specific dependencies

---

**Last Updated:** January 2026

