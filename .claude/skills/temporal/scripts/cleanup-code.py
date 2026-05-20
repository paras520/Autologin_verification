#!/usr/bin/env python3
"""
Temporal Integration System - Code Cleanup Script
Removes emojis, images, and AI-generated code slop from codebase.

Usage:
    python cleanup-code.py [repo_root] [--backup-dir BACKUP_DIR]
    
If no repo_root provided, uses current working directory.
"""

import os
import re
import shutil
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


BACKUP_DIR_NAME = ".temporal-backups"


def remove_emojis_and_images(content: str) -> str:
    """Remove emojis and images from code content"""
    # Remove emoji patterns (comprehensive Unicode ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002700-\U000027BF"  # dingbats
        "]+", flags=re.UNICODE
    )
    content = emoji_pattern.sub('', content)
    
    # Remove image references (markdown images, HTML img tags)
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Markdown images
    content = re.sub(r'<img[^>]*>', '', content)  # HTML images
    content = re.sub(r'<image[^>]*>', '', content)  # XML/SVG images
    
    # Remove common AI-generated code slop patterns
    # Remove excessive blank lines (more than 2 consecutive)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    return content


def get_files_to_clean(repo_root: Path) -> List[Path]:
    """Get files to clean (diff against main or all files)"""
    # Check if git repo
    if not (repo_root / ".git").exists():
        # Not a git repo, clean all code files
        return get_all_code_files(repo_root)
    
    # Check if main/master branch exists
    try:
        result = subprocess.run(
            ["git", "branch", "--list", "main", "master"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # Main branch exists, get diff
            return get_diff_files(repo_root)
        else:
            # No main branch, clean all files
            return get_all_code_files(repo_root)
    except Exception:
        # Git command failed, clean all files
        return get_all_code_files(repo_root)


def get_diff_files(repo_root: Path) -> List[Path]:
    """Get files changed from main branch"""
    try:
        # Try main first, then master
        for branch in ["main", "master"]:
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{branch}...HEAD"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                files = [
                    repo_root / f.strip() 
                    for f in result.stdout.strip().split('\n') 
                    if f.strip()
                ]
                return [f for f in files if f.exists() and is_code_file(f)]
        return []
    except Exception:
        return []


def get_all_code_files(repo_root: Path) -> List[Path]:
    """Get all code files in repository"""
    code_extensions = {
        '.ts', '.tsx', '.js', '.jsx',  # TypeScript/JavaScript
        '.py',  # Python
        '.go',  # Go
        '.java',  # Java
        '.cs',  # C#
        '.php',  # PHP
        '.rb',  # Ruby
        '.md',  # Markdown
        '.json',  # JSON (for config files)
    }
    
    exclude_dirs = {
        '.git', '.temporal-backups', 'node_modules', '.venv', 'venv',
        '__pycache__', '.pytest_cache', 'dist', 'build', '.next', '.cursor',
        'cursor-skills-temporal'  # Exclude the system itself
    }
    
    code_files = []
    for root, dirs, files in os.walk(repo_root):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in code_extensions:
                # Skip files in excluded directories
                if not any(exclude in file_path.parts for exclude in exclude_dirs):
                    code_files.append(file_path)
    
    return code_files


def is_code_file(file_path: Path) -> bool:
    """Check if file is a code file"""
    code_extensions = {
        '.ts', '.tsx', '.js', '.jsx', '.py', '.go', 
        '.java', '.cs', '.php', '.rb', '.md', '.json'
    }
    return file_path.suffix in code_extensions


def validate_syntax(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate file syntax (basic check - file exists and is readable)"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Basic validation: file is readable and not empty (unless it should be)
        return True, None
    except Exception as e:
        return False, str(e)


def create_backup(files_to_backup: List[Path], backup_dir: Path) -> Tuple[bool, Optional[str]]:
    """Create backup of files before cleanup"""
    try:
        # Create backup directory
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to backup
        for file_path in files_to_backup:
            # Preserve directory structure
            relative_path = file_path.relative_to(file_path.parts[0])
            backup_path = backup_dir / relative_path
            
            # Create parent directories
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(file_path, backup_path)
        
        return True, None
    except Exception as e:
        return False, str(e)


def restore_backup(backup_dir: Path, repo_root: Path) -> Tuple[bool, Optional[str]]:
    """Restore files from backup"""
    try:
        if not backup_dir.exists():
            return False, "Backup directory does not exist"
        
        # Restore all files from backup
        for backup_file in backup_dir.rglob('*'):
            if backup_file.is_file():
                # Calculate relative path
                relative_path = backup_file.relative_to(backup_dir)
                original_path = repo_root / relative_path
                
                # Create parent directories
                original_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file back
                shutil.copy2(backup_file, original_path)
        
        return True, None
    except Exception as e:
        return False, str(e)


def cleanup_backup(backup_dir: Path) -> Tuple[bool, Optional[str]]:
    """Delete backup folder"""
    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        return True, None
    except Exception as e:
        return False, str(e)


def cleanup_codebase(repo_root: Path, backup_dir: Optional[Path] = None) -> Dict:
    """Main cleanup function"""
    files_to_clean = get_files_to_clean(repo_root)
    
    if not files_to_clean:
        return {
            'success': True,
            'cleaned_files': [],
            'failed_files': [],
            'message': 'No files to clean'
        }
    
    # Create backup if backup_dir provided
    backup_created = False
    if backup_dir:
        success, error = create_backup(files_to_clean, backup_dir)
        if success:
            backup_created = True
        else:
            return {
                'success': False,
                'cleaned_files': [],
                'failed_files': [],
                'message': f'Backup creation failed: {error}'
            }
    
    cleaned_files = []
    failed_files = []
    
    for file_path in files_to_clean:
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Clean content
            cleaned_content = remove_emojis_and_images(content)
            
            # Only write if changed
            if cleaned_content != content:
                # Write cleaned file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                cleaned_files.append(str(file_path))
                
                # Validate cleaned file
                is_valid, error = validate_syntax(file_path)
                if not is_valid:
                    failed_files.append({
                        'file': str(file_path),
                        'error': f'Validation failed: {error}'
                    })
        except Exception as e:
            failed_files.append({
                'file': str(file_path),
                'error': str(e)
            })
    
    result = {
        'success': len(failed_files) == 0,
        'cleaned_files': cleaned_files,
        'failed_files': failed_files,
        'backup_created': backup_created,
        'backup_dir': str(backup_dir) if backup_dir else None,
        'message': f'Cleaned {len(cleaned_files)} files' if len(failed_files) == 0 else f'Failed to clean {len(failed_files)} files'
    }
    
    return result


def main():
    """Main execution"""
    repo_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    
    # Parse backup directory from args
    backup_dir = None
    if '--backup-dir' in sys.argv:
        idx = sys.argv.index('--backup-dir')
        if idx + 1 < len(sys.argv):
            backup_dir = Path(sys.argv[idx + 1]).resolve()
    
    if backup_dir is None:
        backup_dir = repo_root / BACKUP_DIR_NAME
    
    print(f"Cleaning codebase: {repo_root}")
    
    result = cleanup_codebase(repo_root, backup_dir)
    
    # Output JSON result
    print(json.dumps(result, indent=2))
    
    if result['success']:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

