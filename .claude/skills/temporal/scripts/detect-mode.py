#!/usr/bin/env python3
"""
Temporal Integration System - Mode Detector
Detects which automation mode to use based on codebase state

Usage:
    python detect-mode.py --repo-root /path/to/repo --output mode.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional


def check_registry_exists(repo_root: str) -> tuple[bool, Optional[str]]:
    """
    Check if temporal-registry.json exists
    Returns: (exists, path)
    """
    # Look for registry in multiple possible locations
    possible_paths = [
        os.path.join(repo_root, 'cursor-skills-temporal', 'progress', 'temporal-registry.json'),
        os.path.join(repo_root, 'progress', 'temporal-registry.json'),
        os.path.join(repo_root, '.temporal', 'registry.json'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return (True, path)
    
    return (False, None)


def check_processed_by_agent(registry_path: str) -> bool:
    """
    Check if registry has processed_by_agent flag
    Returns: True if processed by agent, False otherwise
    """
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Check for agent flag
        if registry.get('processed_by_agent', False):
            return True
        
        # Check metadata
        metadata = registry.get('metadata', {})
        if metadata.get('processed_by_agent', False):
            return True
        
        return False
    
    except Exception as e:
        print(f"Warning: Could not read registry: {e}")
        return False


def count_implementations(registry_path: str) -> int:
    """Count number of implementations in registry"""
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        count = 0
        count += len(registry.get('workflows', []))
        count += len(registry.get('activities', []))
        count += len(registry.get('workers', []))
        
        return count
    
    except:
        return 0


def count_apis_in_code(repo_root: str) -> int:
    """
    Quick count of APIs in codebase
    Returns rough estimate of total APIs
    """
    # This is a simplified version - in reality, would call scan-codebase.py
    # For now, check common patterns
    
    count = 0
    api_patterns = [
        '**/*controller*.py',
        '**/*controller*.ts',
        '**/*route*.py',
        '**/*route*.ts',
        '**/*api*.py',
        '**/*api*.ts',
        '**/resolvers/*.py',
        '**/resolvers/*.ts',
    ]
    
    for pattern in api_patterns:
        matches = list(Path(repo_root).glob(pattern))
        # Exclude node_modules, .git, etc.
        matches = [m for m in matches if 'node_modules' not in str(m) and '.git' not in str(m)]
        count += len(matches)
    
    # Rough estimate: each file has ~2-3 APIs on average
    return count * 2


def check_code_exists(repo_root: str) -> bool:
    """
    Check if significant code exists in repository
    Returns: True if code exists, False if empty/minimal
    """
    # Check for common project files
    project_indicators = [
        'package.json',
        'requirements.txt',
        'go.mod',
        'pom.xml',
        'Gemfile',
        'composer.json',
        'Cargo.toml',
    ]
    
    for indicator in project_indicators:
        if os.path.exists(os.path.join(repo_root, indicator)):
            return True
    
    # Check for code directories
    code_dirs = ['src', 'lib', 'app', 'api', 'controllers', 'routes', 'services']
    for dirname in code_dirs:
        dir_path = os.path.join(repo_root, dirname)
        if os.path.isdir(dir_path):
            # Check if it has files
            files = list(Path(dir_path).rglob('*.py')) + \
                   list(Path(dir_path).rglob('*.ts')) + \
                   list(Path(dir_path).rglob('*.js')) + \
                   list(Path(dir_path).rglob('*.go')) + \
                   list(Path(dir_path).rglob('*.java'))
            
            if len(files) > 5:  # More than 5 code files = significant code
                return True
    
    return False


def detect_mode(repo_root: str) -> Dict:
    """
    Detect automation mode based on codebase state
    Returns mode info dict
    """
    # Check registry
    registry_exists, registry_path = check_registry_exists(repo_root)
    
    # Check code
    code_exists = check_code_exists(repo_root)
    
    # Initialize result
    result = {
        'registry_exists': registry_exists,
        'code_exists': code_exists,
        'processed_by_agent': False,
        'implemented_count': 0,
        'total_apis_in_code': 0
    }
    
    # Check agent flag if registry exists
    if registry_exists:
        result['processed_by_agent'] = check_processed_by_agent(registry_path)
        result['implemented_count'] = count_implementations(registry_path)
    
    # Count APIs if code exists
    if code_exists:
        result['total_apis_in_code'] = count_apis_in_code(repo_root)
    
    # Determine mode
    if not code_exists or result['total_apis_in_code'] < 5:
        # New module: no significant code yet
        mode = "new-module"
        explanation = "No significant code detected. Will auto-implement as you write code."
    
    elif registry_exists and result['processed_by_agent']:
        # Post-agent: Registry exists with agent flag
        mode = "post-agent"
        remaining = result['total_apis_in_code'] - result['implemented_count']
        explanation = f"Agent already processed this repo ({result['implemented_count']} items). " \
                     f"{remaining} items remaining (likely skipped as LOW priority)."
    
    elif code_exists and not registry_exists:
        # First-time existing: Code exists but no registry
        mode = "first-time-existing"
        explanation = f"Existing codebase detected ({result['total_apis_in_code']} APIs estimated). " \
                     f"Will ask once which to implement, then auto-implement future code."
    
    elif registry_exists and not result['processed_by_agent']:
        # Registry exists but not from agent (manual/interactive mode)
        # Treat as post-agent but with different message
        mode = "post-agent"
        remaining = result['total_apis_in_code'] - result['implemented_count']
        explanation = f"Registry exists ({result['implemented_count']} items implemented). " \
                     f"{remaining} items remaining. Will auto-implement new code."
    
    else:
        # Default to new-module
        mode = "new-module"
        explanation = "Unable to determine state. Will auto-implement as code is written."
    
    result['mode'] = mode
    result['explanation'] = explanation
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Detect automation mode for Temporal system')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    print(f"Detecting mode for: {args.repo_root}")
    
    # Detect mode
    result = detect_mode(args.repo_root)
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Print result
    print(f"\n{'='*60}")
    print(f"Mode Detection Result")
    print(f"{'='*60}")
    print(f"Mode: {result['mode'].upper()}")
    print(f"\n{result['explanation']}")
    print(f"\nDetails:")
    print(f"  - Registry exists: {result['registry_exists']}")
    print(f"  - Code exists: {result['code_exists']}")
    print(f"  - Processed by agent: {result['processed_by_agent']}")
    print(f"  - Implemented count: {result['implemented_count']}")
    print(f"  - Total APIs in code: {result['total_apis_in_code']}")
    print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
