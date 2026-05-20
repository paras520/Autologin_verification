#!/usr/bin/env python3
"""
Temporal Integration System - Priority Analyzer
Analyzes APIs/services and assigns priority scores (HIGH/LOW)

Usage:
    python priority-analyzer.py --scan-results <scan-results.json> --output <priorities.json>
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Priority scoring factors
SCORE_EXTERNAL_API = 30
SCORE_LONG_RUNNING = 25
SCORE_MULTI_STEP = 20
SCORE_NEEDS_RETRY = 15
SCORE_COMPLEX_STATE = 10

# Threshold for HIGH priority
HIGH_PRIORITY_THRESHOLD = 50


# External API patterns to detect
EXTERNAL_API_PATTERNS = [
    # API service names
    r'\b(openai|stripe|aws|twilio|sendgrid|mailgun|firebase|mongodb|redis|postgresql)\b',
    # HTTP clients
    r'\b(axios|fetch|http|request|RestClient|HttpClient|WebClient)\b',
    # API operations
    r'\bapi\.(get|post|put|delete|patch)',
    # External service calls
    r'\b(payment|email|sms|storage|database|cache|queue)\b'
]


# Long-running operation patterns
LONG_RUNNING_PATTERNS = [
    r'\bsleep\(',
    r'\bawait\s+\w+\(',
    r'\bsetTimeout\(',
    r'\btime\.sleep\(',
    r'\bThread\.sleep\(',
    r'\bTask\.Delay\(',
    r'\bfor\s+\w+\s+in\s+range\(',
    r'\bwhile\s+\w+',
    r'\b(batch|bulk|process|generate|convert|transform|migrate)\b'
]


# Multi-step workflow patterns
MULTI_STEP_PATTERNS = [
    r'\bstep\s*\d+',
    r'\bphase\s*\d+',
    r'\bthen\s+',
    r'\bnext\s+',
    r'workflow',
    r'pipeline',
    r'orchestrat',
    r'sequenc',
    r'chain'
]


# Retry/error-prone patterns
RETRY_PATTERNS = [
    r'\bretry',
    r'\bcatch\s*\(',
    r'\bexcept\s+',
    r'\brescue\s+',
    r'\berror\s+handling',
    r'\bfallback',
    r'\btimeout',
    r'\bbackoff',
    r'\bresilience'
]


# Complex state patterns
STATE_PATTERNS = [
    r'\bstate\s*=',
    r'\bself\.\w+\s*=',
    r'\bthis\.\w+\s*=',
    r'\bprivate\s+\w+',
    r'\bprotected\s+\w+',
    r'\bstatic\s+\w+',
    r'\bdatabase\.',
    r'\bstore\.',
    r'\bcache\.',
    r'\bsession\.'
]


def read_file_content(file_path: str) -> str:
    """Read file content safely"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""


def detect_external_api_calls(item: Dict, code_content: str = "") -> Tuple[bool, int]:
    """
    Detect if item uses external APIs
    Returns: (has_external_apis, score)
    """
    score = 0
    
    # Check item metadata
    item_type = item.get('type', '')
    item_name = item.get('name', '').lower()
    item_path = item.get('path', '').lower()
    
    # Check if it's already marked as external API
    if item_type == 'external_api' or 'api' in item.get('category', '').lower():
        score += SCORE_EXTERNAL_API
        return (True, score)
    
    # Check name/path for API indicators
    for pattern in EXTERNAL_API_PATTERNS:
        if re.search(pattern, item_name, re.IGNORECASE):
            score += SCORE_EXTERNAL_API
            return (True, score)
        if re.search(pattern, item_path, re.IGNORECASE):
            score += SCORE_EXTERNAL_API
            return (True, score)
    
    # Check code content if available
    if code_content:
        matches = 0
        for pattern in EXTERNAL_API_PATTERNS:
            if re.search(pattern, code_content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:  # Multiple API indicators
            score += SCORE_EXTERNAL_API
            return (True, score)
    
    return (False, 0)


def detect_long_running(item: Dict, code_content: str = "") -> Tuple[bool, int]:
    """
    Detect if operation is long-running
    Returns: (is_long_running, score)
    """
    score = 0
    
    item_name = item.get('name', '').lower()
    
    # Check name for indicators
    long_running_keywords = ['batch', 'bulk', 'process', 'generate', 'convert', 'migrate', 'export', 'import']
    if any(keyword in item_name for keyword in long_running_keywords):
        score += SCORE_LONG_RUNNING
        return (True, score)
    
    # Check code content
    if code_content:
        matches = 0
        for pattern in LONG_RUNNING_PATTERNS:
            if re.search(pattern, code_content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            score += SCORE_LONG_RUNNING
            return (True, score)
    
    return (False, 0)


def detect_multi_step(item: Dict, code_content: str = "") -> Tuple[bool, int]:
    """
    Detect if it's a multi-step workflow
    Returns: (is_multi_step, score)
    """
    score = 0
    
    item_name = item.get('name', '').lower()
    
    # Check name
    multi_step_keywords = ['workflow', 'pipeline', 'orchestrate', 'process', 'sequence']
    if any(keyword in item_name for keyword in multi_step_keywords):
        score += SCORE_MULTI_STEP
        return (True, score)
    
    # Check code content
    if code_content:
        matches = 0
        for pattern in MULTI_STEP_PATTERNS:
            if re.search(pattern, code_content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            score += SCORE_MULTI_STEP
            return (True, score)
    
    return (False, 0)


def detect_retry_needs(item: Dict, code_content: str = "") -> Tuple[bool, int]:
    """
    Detect if operation needs retry logic
    Returns: (needs_retry, score)
    """
    score = 0
    
    item_name = item.get('name', '').lower()
    
    # Check name for error-prone operations
    error_prone = ['payment', 'transaction', 'order', 'booking', 'reservation', 'submit']
    if any(keyword in item_name for keyword in error_prone):
        score += SCORE_NEEDS_RETRY
        return (True, score)
    
    # Check code content
    if code_content:
        matches = 0
        for pattern in RETRY_PATTERNS:
            if re.search(pattern, code_content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            score += SCORE_NEEDS_RETRY
            return (True, score)
    
    return (False, 0)


def detect_complex_state(item: Dict, code_content: str = "") -> Tuple[bool, int]:
    """
    Detect if operation has complex state management
    Returns: (has_complex_state, score)
    """
    score = 0
    
    # Check code content
    if code_content:
        matches = 0
        for pattern in STATE_PATTERNS:
            if re.search(pattern, code_content, re.IGNORECASE):
                matches += 1
        
        if matches >= 3:
            score += SCORE_COMPLEX_STATE
            return (True, score)
    
    return (False, 0)


def analyze_item(item: Dict, repo_root: str = ".") -> Dict:
    """
    Analyze a single API/service and return priority info
    """
    # Read code content if file path available
    code_content = ""
    if 'file' in item:
        file_path = os.path.join(repo_root, item['file'])
        code_content = read_file_content(file_path)
    elif 'files' in item and item['files']:
        # Read first file
        file_path = os.path.join(repo_root, item['files'][0])
        code_content = read_file_content(file_path)
    
    # Score each factor
    factors = {}
    total_score = 0
    
    has_external, external_score = detect_external_api_calls(item, code_content)
    factors['external_api_calls'] = external_score
    total_score += external_score
    
    is_long, long_score = detect_long_running(item, code_content)
    factors['long_running'] = long_score
    total_score += long_score
    
    is_multi, multi_score = detect_multi_step(item, code_content)
    factors['multi_step'] = multi_score
    total_score += multi_score
    
    needs_retry, retry_score = detect_retry_needs(item, code_content)
    factors['needs_retry'] = retry_score
    total_score += retry_score
    
    has_state, state_score = detect_complex_state(item, code_content)
    factors['complex_state'] = state_score
    total_score += state_score
    
    # Determine priority
    priority = "HIGH" if total_score >= HIGH_PRIORITY_THRESHOLD else "LOW"
    
    # Generate explanation
    explanation = generate_explanation(item, factors, priority)
    
    return {
        'name': item.get('name', item.get('endpoint', 'unknown')),
        'score': total_score,
        'priority': priority,
        'factors': factors,
        'explanation': explanation
    }


def generate_explanation(item: Dict, factors: Dict, priority: str) -> str:
    """Generate human-readable explanation for priority score"""
    reasons = []
    
    if factors['external_api_calls'] > 0:
        reasons.append("uses external APIs")
    if factors['long_running'] > 0:
        reasons.append("long-running operation")
    if factors['multi_step'] > 0:
        reasons.append("multi-step workflow")
    if factors['needs_retry'] > 0:
        reasons.append("needs retry logic")
    if factors['complex_state'] > 0:
        reasons.append("complex state management")
    
    item_name = item.get('name', item.get('endpoint', 'this item'))
    
    if priority == "HIGH":
        if reasons:
            return f"High priority: {item_name} - {', '.join(reasons)}"
        else:
            return f"High priority: {item_name}"
    else:
        if reasons:
            return f"Low priority: {item_name} - {', '.join(reasons)}"
        else:
            return f"Low priority: {item_name} - simple operation"


def analyze_scan_results(scan_results: Dict, repo_root: str = ".") -> Dict:
    """
    Analyze all APIs/services from scan results
    Returns dict with priority info for each item
    """
    priorities = {}
    
    # Analyze internal APIs
    internal_apis = scan_results.get('internal_apis', [])
    for i, api in enumerate(internal_apis):
        api_id = f"internal_api_{i}"
        priorities[api_id] = analyze_item(api, repo_root)
    
    # Analyze external APIs
    external_apis = scan_results.get('external_apis', [])
    for i, api in enumerate(external_apis):
        api_id = f"external_api_{i}"
        priorities[api_id] = analyze_item(api, repo_root)
    
    # Analyze services
    services = scan_results.get('services', [])
    for i, service in enumerate(services):
        service_id = f"service_{i}"
        priorities[service_id] = analyze_item(service, repo_root)
    
    return priorities


def get_priority_summary(priorities: Dict) -> Dict:
    """Generate summary statistics"""
    high_count = sum(1 for p in priorities.values() if p['priority'] == 'HIGH')
    low_count = sum(1 for p in priorities.values() if p['priority'] == 'LOW')
    total = len(priorities)
    
    avg_score = sum(p['score'] for p in priorities.values()) / total if total > 0 else 0
    
    return {
        'total_items': total,
        'high_priority': high_count,
        'low_priority': low_count,
        'average_score': round(avg_score, 2)
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze priorities for Temporal implementations')
    parser.add_argument('--scan-results', required=True, help='Path to scan results JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file for priorities')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    
    args = parser.parse_args()
    
    print(f"Analyzing priorities from: {args.scan_results}")
    
    # Load scan results
    try:
        with open(args.scan_results, 'r') as f:
            scan_results = json.load(f)
    except Exception as e:
        print(f"Error loading scan results: {e}")
        return 1
    
    # Analyze priorities
    priorities = analyze_scan_results(scan_results, args.repo_root)
    
    # Generate summary
    summary = get_priority_summary(priorities)
    
    # Prepare output
    output = {
        'summary': summary,
        'priorities': priorities
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Priority Analysis Summary")
    print(f"{'='*60}")
    print(f"Total Items: {summary['total_items']}")
    print(f"High Priority: {summary['high_priority']} ({summary['high_priority']/summary['total_items']*100:.0f}%)")
    print(f"Low Priority: {summary['low_priority']} ({summary['low_priority']/summary['total_items']*100:.0f}%)")
    print(f"Average Score: {summary['average_score']}")
    print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
