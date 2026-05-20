#!/usr/bin/env python3
"""
Temporal Integration System - Validation Script
Validates system integrity before distribution.

Usage:
    python validate-system.py
    
Runs from cursor-skills-temporal/scripts/ directory.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class ValidationResult:
    """Stores validation results"""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, check: str):
        self.passed.append(check)
    
    def add_fail(self, check: str, reason: str = ""):
        self.failed.append((check, reason))
    
    def add_warning(self, check: str, reason: str = ""):
        self.warnings.append((check, reason))
    
    def is_success(self) -> bool:
        return len(self.failed) == 0
    
    def print_results(self):
        """Print validation results"""
        print("\n" + "="*60)
        print("  VALIDATION RESULTS")
        print("="*60 + "\n")
        
        if self.passed:
            print(f"[OK] PASSED ({len(self.passed)} checks):")
            for check in self.passed:
                print(f"   [+] {check}")
            print()
        
        if self.warnings:
            print(f"[!] WARNINGS ({len(self.warnings)}):")
            for check, reason in self.warnings:
                print(f"   [!] {check}")
                if reason:
                    print(f"      -> {reason}")
            print()
        
        if self.failed:
            print(f"[X] FAILED ({len(self.failed)} checks):")
            for check, reason in self.failed:
                print(f"   [X] {check}")
                if reason:
                    print(f"      -> {reason}")
            print()
        
        print("="*60)
        if self.is_success():
            print("[OK] ALL VALIDATION CHECKS PASSED")
            print("[OK] System ready for distribution")
        else:
            print("[X] VALIDATION FAILED")
            print("[X] Fix errors above before distributing")
        print("="*60 + "\n")


def find_base_folder() -> Path:
    """Find cursor-skills-temporal folder"""
    script_path = Path(__file__).resolve()
    # Script is in cursor-skills-temporal/scripts/validate-system.py
    # So parent.parent is cursor-skills-temporal/
    base_folder = script_path.parent.parent
    return base_folder


def validate_folder_structure(base_folder: Path, result: ValidationResult):
    """Validate folder structure"""
    print("[1/10] Validating folder structure...")
    
    required_folders = [
        "for-cursor",
        "for-cursor/references",
        "for-developer",
        "progress",
        "scripts"
    ]
    
    all_exist = True
    for folder in required_folders:
        folder_path = base_folder / folder
        if folder_path.exists() and folder_path.is_dir():
            result.add_pass(f"Folder exists: {folder}")
        else:
            result.add_fail(f"Folder missing: {folder}")
            all_exist = False
    
    if all_exist:
        print("   [+] All required folders present")
    else:
        print("   [X] Some folders missing")


def validate_required_files(base_folder: Path, result: ValidationResult):
    """Validate all required files exist"""
    print("[2/10] Validating required files...")
    
    required_files = {
        # Root files
        ".cursorrules": "Main configuration file",
        "CODE_FLOW.md": "System flow documentation",
        "CURRENT_STATE.md": "System status tracker",
        
        # for-cursor files
        "for-cursor/summary.md": "System state tracker",
        "for-cursor/temporal-overview.mdc": "Platform overview",
        "for-cursor/temporal-workflows.mdc": "Workflow patterns",
        "for-cursor/temporal-activities.mdc": "Activity patterns",
        "for-cursor/temporal-workers.mdc": "Worker patterns",
        "for-cursor/temporal-testing.mdc": "Testing strategies",
        "for-cursor/temporal-deployment.mdc": "Deployment patterns",
        "for-cursor/temporal-use-cases.mdc": "Use case examples",
        "for-cursor/temporal-error-handling.mdc": "Error handling",
        "for-cursor/temporal-language-examples.mdc": "Language examples",
        
        # for-cursor/references files
        "for-cursor/references/discovery-protocol.md": "Discovery protocol",
        "for-cursor/references/execution-protocol.md": "Execution protocol",
        "for-cursor/references/validation-protocol.md": "Validation protocol",
        "for-cursor/references/pattern-detection.md": "Pattern detection",
        "for-cursor/references/implementation-guidelines.md": "Implementation guidelines",
        
        # for-developer files
        "for-developer/README.md": "Complete documentation",
        "for-developer/DEVELOPER_SETUP_3_STEPS.md": "Setup guide",
        "for-developer/DEVELOPER_GUIDE.md": "System documentation",
        "for-developer/ENV_SETUP_GUIDE.md": "Environment setup",
        "for-developer/REGISTRY_GUIDE.md": "Registry guide",
        "for-developer/VERSION_CHECKING_GUIDE.md": "Version checking",
        "for-developer/MAINTENANCE_GUIDE.md": "Maintenance guide",
        
        # progress files
        "progress/temporal-registry.json": "Implementation registry",
        
        # scripts
        "scripts/setup-temporal.py": "Setup script",
        "scripts/validate-system.py": "Validation script",
        "scripts/scan-codebase.py": "Codebase scanner",
        "scripts/refactor-to-temporal.py": "Refactoring engine",
        "scripts/cleanup-code.py": "Code cleanup script",
        "scripts/generate-tests.py": "Test generator",
        "scripts/run-tests.py": "Test runner",
        "scripts/create-test-reports.py": "Test report generator",
        "scripts/detect-mode.py": "Mode detector",
        "scripts/priority-analyzer.py": "Priority analyzer",
        "scripts/deep-scanner.py": "Deep scanner",
        "scripts/USAGE.md": "Scripts usage guide"
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        full_path = base_folder / file_path
        if full_path.exists() and full_path.is_file():
            result.add_pass(f"File exists: {file_path}")
        else:
            result.add_fail(f"File missing: {file_path}", description)
            all_exist = False
    
    if all_exist:
        print(f"   [+] All {len(required_files)} required files present")
    else:
        print(f"   [X] Some files missing")


def validate_cursorrules(base_folder: Path, result: ValidationResult):
    """Validate .cursorrules file"""
    print("[3/10] Validating .cursorrules file...")
    
    cursorrules_path = base_folder / ".cursorrules"
    
    if not cursorrules_path.exists():
        result.add_fail(".cursorrules file missing")
        return
    
    try:
        with open(cursorrules_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            "DYNAMIC FOLDER DISCOVERY",
            "AUTOMATIC EXECUTION PROTOCOL",
            "CONFIGURATION VALIDATION",
            "PATTERN DETECTION",
            "IMPLEMENTATION GUIDELINES",
            "KNOWLEDGE BASE READING ORDER",
            "STARTUP CODEBASE ANALYSIS",
            "INTERACTIVE APPROVAL WORKFLOW",
            "POST-IMPLEMENTATION CLEANUP"
        ]
        
        all_sections_present = True
        for section in required_sections:
            if section in content:
                result.add_pass(f".cursorrules contains: {section}")
            else:
                result.add_fail(f".cursorrules missing section: {section}")
                all_sections_present = False
        
        # Check references to protocol files
        if "discovery-protocol.md" in content:
            result.add_pass(".cursorrules references discovery-protocol.md")
        else:
            result.add_warning(".cursorrules doesn't reference discovery-protocol.md")
        
        if all_sections_present:
            print("   [+] .cursorrules file valid")
        else:
            print("   [X] .cursorrules file incomplete")
            
    except Exception as e:
        result.add_fail(".cursorrules file validation failed", str(e))
        print(f"   [X] Error reading .cursorrules: {e}")


def validate_json_files(base_folder: Path, result: ValidationResult):
    """Validate JSON files are valid"""
    print("[4/10] Validating JSON files...")
    
    json_files = [
        "progress/temporal-registry.json"
    ]
    
    all_valid = True
    for json_file in json_files:
        file_path = base_folder / json_file
        
        if not file_path.exists():
            result.add_fail(f"JSON file missing: {json_file}")
            all_valid = False
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate registry structure
            if json_file == "progress/temporal-registry.json":
                required_keys = ["version", "implementations", "files", "statistics", "metadata"]
                for key in required_keys:
                    if key in data:
                        result.add_pass(f"Registry contains: {key}")
                    else:
                        result.add_fail(f"Registry missing key: {key}")
                        all_valid = False
            else:
                result.add_pass(f"Valid JSON: {json_file}")
                
        except json.JSONDecodeError as e:
            result.add_fail(f"Invalid JSON: {json_file}", str(e))
            all_valid = False
        except Exception as e:
            result.add_fail(f"Error reading JSON: {json_file}", str(e))
            all_valid = False
    
    if all_valid:
        print("   [+] All JSON files valid")
    else:
        print("   [X] Some JSON files invalid")


def validate_mdc_files(base_folder: Path, result: ValidationResult):
    """Validate .mdc knowledge base files"""
    print("[5/10] Validating knowledge base (.mdc) files...")
    
    mdc_files = [
        "temporal-overview.mdc",
        "temporal-workflows.mdc",
        "temporal-activities.mdc",
        "temporal-workers.mdc",
        "temporal-testing.mdc",
        "temporal-deployment.mdc",
        "temporal-use-cases.mdc",
        "temporal-error-handling.mdc",
        "temporal-language-examples.mdc"
    ]
    
    all_exist = True
    for mdc_file in mdc_files:
        file_path = base_folder / "for-cursor" / mdc_file
        if file_path.exists() and file_path.is_file():
            # Check file is not empty
            if file_path.stat().st_size > 0:
                result.add_pass(f"Knowledge base file: {mdc_file}")
            else:
                result.add_warning(f"Empty file: {mdc_file}", "File exists but is empty")
                all_exist = False
        else:
            result.add_fail(f"Missing knowledge base file: {mdc_file}")
            all_exist = False
    
    if all_exist:
        print(f"   [+] All {len(mdc_files)} .mdc files present")
    else:
        print("   [X] Some .mdc files missing or empty")


def validate_reference_files(base_folder: Path, result: ValidationResult):
    """Validate reference protocol files"""
    print("[6/10] Validating reference protocol files...")
    
    reference_files = [
        "discovery-protocol.md",
        "execution-protocol.md",
        "validation-protocol.md",
        "pattern-detection.md",
        "implementation-guidelines.md"
    ]
    
    all_exist = True
    for ref_file in reference_files:
        file_path = base_folder / "for-cursor" / "references" / ref_file
        if file_path.exists() and file_path.is_file():
            if file_path.stat().st_size > 0:
                result.add_pass(f"Reference file: {ref_file}")
            else:
                result.add_fail(f"Empty reference file: {ref_file}")
                all_exist = False
        else:
            result.add_fail(f"Missing reference file: {ref_file}")
            all_exist = False
    
    if all_exist:
        print(f"   [+] All {len(reference_files)} reference files present")
    else:
        print("   [X] Some reference files missing")


def validate_scripts(base_folder: Path, result: ValidationResult):
    """Validate scripts are executable"""
    print("[7/10] Validating scripts...")
    
    scripts = [
        "scripts/setup-temporal.py",
        "scripts/validate-system.py",
        "scripts/scan-codebase.py",
        "scripts/refactor-to-temporal.py",
        "scripts/cleanup-code.py",
        "scripts/generate-tests.py",
        "scripts/run-tests.py",
        "scripts/create-test-reports.py",
        "scripts/detect-mode.py",
        "scripts/priority-analyzer.py",
        "scripts/deep-scanner.py"
    ]
    
    all_valid = True
    for script in scripts:
        script_path = base_folder / script
        if script_path.exists():
            # Check if file has shebang
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                if first_line.startswith('#!/'):
                    result.add_pass(f"Script valid: {script}")
                else:
                    result.add_warning(f"Script missing shebang: {script}")
            except Exception as e:
                result.add_fail(f"Error reading script: {script}", str(e))
                all_valid = False
        else:
            result.add_fail(f"Script missing: {script}")
            all_valid = False
    
    if all_valid:
        print(f"   [+] All scripts valid")
    else:
        print("   [X] Some scripts invalid")


def count_files(base_folder: Path, result: ValidationResult):
    """Count total files and verify expected count"""
    print("[8/10] Counting files...")
    
    # Expected file count (production files only, excludes development planning docs)
    expected_count = 27  # All essential system files
    
    # Count all files recursively
    all_files = []
    for root, dirs, files in os.walk(base_folder):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if not file.startswith('.'):
                rel_path = Path(root).relative_to(base_folder) / file
                all_files.append(str(rel_path))
    
    actual_count = len(all_files)
    
    if actual_count >= expected_count:
        result.add_pass(f"File count: {actual_count} files (expected ~{expected_count})")
        print(f"   [+] Found {actual_count} files")
    else:
        result.add_warning(f"File count: {actual_count} files (expected ~{expected_count})", 
                          f"Missing {expected_count - actual_count} files")
        print(f"   [!] Found {actual_count} files (expected ~{expected_count})")


def validate_naming_conventions(base_folder: Path, result: ValidationResult):
    """Validate file naming conventions"""
    print("[9/10] Validating naming conventions...")
    
    conventions = {
        ".cursorrules": "Must be exactly '.cursorrules' (no extension)",
        "summary.md": "Must be 'summary.md' (lowercase)",
        "temporal-registry.json": "Must be 'temporal-registry.json' (lowercase with hyphens)"
    }
    
    all_correct = True
    for file_name, description in conventions.items():
        # Check if file exists with correct name
        found = False
        for root, dirs, files in os.walk(base_folder):
            if file_name in files:
                found = True
                result.add_pass(f"Naming correct: {file_name}")
                break
        
        if not found:
            result.add_fail(f"File not found or incorrectly named: {file_name}", description)
            all_correct = False
    
    if all_correct:
        print("   [+] All naming conventions correct")
    else:
        print("   [X] Some naming issues found")


def validate_documentation(base_folder: Path, result: ValidationResult):
    """Validate documentation files are not empty and contain expected content"""
    print("[10/10] Validating documentation content...")
    
    doc_files = {
        "for-developer/README.md": "Complete system documentation",
        "for-developer/DEVELOPER_SETUP_3_STEPS.md": "Setup instructions",
        "START_HERE.md": "Quick start guide" if (base_folder.parent / "START_HERE.md").exists() else None
    }
    
    all_valid = True
    for doc_file, description in doc_files.items():
        if description is None:
            continue
            
        file_path = base_folder / doc_file
        if not file_path.exists():
            # Try parent directory for START_HERE.md
            file_path = base_folder.parent / doc_file
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content) > 100:  # Basic check for substantial content
                    result.add_pass(f"Documentation valid: {doc_file}")
                else:
                    result.add_warning(f"Documentation too short: {doc_file}")
                    all_valid = False
            except Exception as e:
                result.add_fail(f"Error reading documentation: {doc_file}", str(e))
                all_valid = False
        else:
            result.add_warning(f"Documentation not found: {doc_file}")
    
    if all_valid:
        print("   [+] All documentation valid")
    else:
        print("   [!] Some documentation issues found")


def main():
    """Main validation function"""
    print("\n" + "="*60)
    print("  Temporal Integration System - Validation")
    print("="*60 + "\n")
    
    # Find base folder
    base_folder = find_base_folder()
    print(f"Validating: {base_folder}\n")
    
    # Create result tracker
    result = ValidationResult()
    
    # Run all validation checks
    validate_folder_structure(base_folder, result)
    validate_required_files(base_folder, result)
    validate_cursorrules(base_folder, result)
    validate_json_files(base_folder, result)
    validate_mdc_files(base_folder, result)
    validate_reference_files(base_folder, result)
    validate_scripts(base_folder, result)
    count_files(base_folder, result)
    validate_naming_conventions(base_folder, result)
    validate_documentation(base_folder, result)
    
    # Print results
    result.print_results()
    
    # Exit with appropriate code
    sys.exit(0 if result.is_success() else 1)


if __name__ == "__main__":
    main()

