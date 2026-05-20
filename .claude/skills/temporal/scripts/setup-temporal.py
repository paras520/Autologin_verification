#!/usr/bin/env python3
"""
Temporal Skill Setup Script
Deploys Temporal skill to a target module and merges .cursorrules
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

def setup_temporal_skill(target_module_path: str, source_skills_path: str = None, rule_mode: str = None):
    """
    Setup Temporal skill in target module
    
    Args:
        target_module_path: Path to target module (e.g., /path/to/m46_FTP_java)
        source_skills_path: Path to skills folder (default: current dir/skills)
        rule_mode: 'file' for .cursorrules file, 'folder' for .cursor/ folder, None for auto-detect
    """
    
    # Resolve paths
    target_path = Path(target_module_path).resolve()
    if not target_path.exists():
        print(f"❌ Target module not found: {target_path}")
        sys.exit(1)
    
    if source_skills_path:
        skills_path = Path(source_skills_path).resolve()
    else:
        # Assume script is in skills/temporal/scripts/
        current_dir = Path(__file__).parent
        skills_path = current_dir.parent.parent  # Go up to skills/
    
    temporal_source = skills_path / "temporal"
    if not temporal_source.exists():
        print(f"❌ Temporal skill not found: {temporal_source}")
        sys.exit(1)
    
    print(f"📦 Deploying Temporal skill to: {target_path}")
    print(f"📁 Source: {temporal_source}")
    print()
    
    # Detect rule mode if not specified
    if rule_mode is None:
        cursor_folder = target_path / ".cursor"
        if cursor_folder.exists() and cursor_folder.is_dir():
            rule_mode = "folder"
            print("🔍 Detected: Module uses .cursor/ folder structure")
        else:
            rule_mode = "file"
            print("🔍 Detected: Module uses .cursorrules file structure")
    else:
        print(f"🔍 Rule mode specified: {rule_mode}")
    
    print()
    
    # Step 1: Copy skills/temporal/ folder to target module
    print("Step 1: Copying skills/temporal/ folder...")
    target_skills = target_path / "skills"
    target_temporal = target_skills / "temporal"
    
    if target_temporal.exists():
        print(f"⚠️  skills/temporal/ already exists. Updating...")
        shutil.rmtree(target_temporal)
    
    target_skills.mkdir(exist_ok=True)
    shutil.copytree(temporal_source, target_temporal)
    print(f"✅ Copied to: {target_temporal}")
    print()
    
    # Step 2: Handle .cursorrules merging (mode-aware)
    print("Step 2: Setting up .cursorrules...")
    
    source_cursorrules = temporal_source / ".cursorrules"
    
    # Read master .cursorrules from root of skills folder
    master_cursorrules_path = skills_path.parent / ".cursorrules"
    if not master_cursorrules_path.exists():
        print(f"⚠️  Master .cursorrules not found at: {master_cursorrules_path}")
        print(f"   Using skill-specific .cursorrules only")
        master_content = ""
    else:
        with open(master_cursorrules_path, 'r', encoding='utf-8') as f:
            master_content = f.read()
    
    # Read skill-specific .cursorrules
    with open(source_cursorrules, 'r', encoding='utf-8') as f:
        skill_content = f.read()
    
    # Combine master + skill content
    combined_content = f"""# Master Cursor Skills Orchestrator
# Auto-deployed by setup script on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{master_content}

---

# Temporal Integration Skill
# Skill-specific rules for Temporal workflow orchestration

{skill_content}
"""
    
    if rule_mode == "folder":
        # MODE: .cursor/ folder
        cursor_folder = target_path / ".cursor"
        cursor_folder.mkdir(exist_ok=True)
        
        target_master_file = cursor_folder / "master-orchestrator.md"
        
        print(f"📁 Module uses .cursor/ folder structure")
        print(f"   Existing rules in .cursor/ will be preserved")
        
        # List existing rules (for info)
        existing_rules = list(cursor_folder.glob("*.md"))
        if existing_rules:
            print(f"   Existing rule files:")
            for rule_file in existing_rules:
                if rule_file.name != "master-orchestrator.md":
                    print(f"     - {rule_file.name} (preserved)")
        
        # Check if master-orchestrator.md already exists
        if target_master_file.exists():
            backup_file = cursor_folder / f"master-orchestrator.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            shutil.copy2(target_master_file, backup_file)
            print(f"   Backup created: {backup_file.name}")
        
        # Write combined content to .cursor/master-orchestrator.md
        with open(target_master_file, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        print(f"✅ Deployed to: .cursor/master-orchestrator.md")
        print(f"   Cursor will read ALL .md files in .cursor/ folder")
        print(f"   Your existing rules + Temporal rules work together ✅")
    
    else:
        # MODE: .cursorrules file
        target_cursorrules = target_path / ".cursorrules"
        
        print(f"📄 Module uses .cursorrules file structure")
        
        if target_cursorrules.exists():
            # Module has existing .cursorrules - smart merge
            print(f"   Existing .cursorrules found")
            backup_cursorrules = target_path / ".cursorrules.backup"
            
            # Backup existing
            shutil.copy2(target_cursorrules, backup_cursorrules)
            print(f"   Backup created: {backup_cursorrules.name}")
            
            # Read existing content
            with open(target_cursorrules, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # Merge: Existing first, then combined (master + skill)
            merged_content = f"""{existing_content}

---

{combined_content}
"""
            
            # Write merged file
            with open(target_cursorrules, 'w', encoding='utf-8') as f:
                f.write(merged_content)
            
            print(f"✅ Merged .cursorrules (existing + master + temporal)")
        
        else:
            # No existing .cursorrules - write combined content
            print(f"   No existing .cursorrules found")
            
            with open(target_cursorrules, 'w', encoding='utf-8') as f:
                f.write(combined_content)
            
            print(f"✅ Created .cursorrules (master + temporal)")
    
    print()
    
    # Step 3: Initialize registry
    print("Step 3: Initializing registry...")
    registry_path = target_temporal / "progress" / "temporal-registry.json"
    
    if registry_path.exists():
        print(f"📋 Registry already exists: {registry_path}")
    else:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        initial_registry = {
            "project": target_path.name,
            "initialized": datetime.now().isoformat(),
            "processed_by_agent": False,
            "implementations": [],
            "skipped_items": []
        }
        
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(initial_registry, f, indent=2)
        
        print(f"✅ Created registry: {registry_path}")
    
    print()
    
    # Step 4: Summary
    print("="*60)
    print("✅ TEMPORAL SKILL DEPLOYMENT COMPLETE")
    print("="*60)
    print()
    print("📁 Deployed to:")
    print(f"   - skills/temporal/ folder: {target_temporal}")
    if rule_mode == "folder":
        print(f"   - Master orchestrator: .cursor/master-orchestrator.md")
        print(f"   - Existing .cursor/*.md rules: Preserved ✅")
    else:
        print(f"   - Rules file: .cursorrules (at root)")
    print(f"   - Registry: {registry_path}")
    print()
    print(f"📋 Rule mode: {rule_mode.upper()}")
    if rule_mode == "folder":
        print(f"   Cursor will read ALL .md files in .cursor/ folder")
        print(f"   Your existing rules work alongside Temporal rules ✅")
    else:
        print(f"   Cursor will read .cursorrules at repository root")
    print()
    print("🚀 Next steps:")
    print("   1. Open module in Cursor IDE")
    print("   2. Cursor will automatically read rules")
    print("   3. System will scan codebase and offer Temporal implementation")
    print()
    print("📖 Documentation:")
    print(f"   - Temporal skill docs: {target_temporal / 'for-developer' / 'README.md'}")
    print(f"   - Skill overview: {target_temporal / 'CODE_FLOW.md'}")
    print()
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup-temporal.py <target_module_path> [source_skills_path] [--rule-mode MODE]")
        print()
        print("Arguments:")
        print("  target_module_path    Path to target module (required)")
        print("  source_skills_path    Path to skills folder (optional, auto-detected if not provided)")
        print("  --rule-mode MODE      'file' for .cursorrules file, 'folder' for .cursor/ folder")
        print("                        (optional, auto-detected if not provided)")
        print()
        print("Examples:")
        print("  python setup-temporal.py /path/to/m46_FTP_java")
        print("  python setup-temporal.py ../../../m46_FTP_java --rule-mode folder")
        print("  python setup-temporal.py C:\\repos\\m46_FTP_java /path/to/skills")
        sys.exit(1)
    
    target = sys.argv[1]
    source = None
    mode = None
    
    # Parse remaining arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--rule-mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            if mode not in ['file', 'folder']:
                print(f"❌ Invalid rule-mode: {mode}. Must be 'file' or 'folder'")
                sys.exit(1)
            i += 2
        else:
            # Assume it's source_skills_path
            source = sys.argv[i]
            i += 1
    
    setup_temporal_skill(target, source, mode)
