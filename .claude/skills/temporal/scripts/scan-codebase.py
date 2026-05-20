#!/usr/bin/env python3
"""
Temporal Integration System - Codebase Scanner (Two-Phase)
Phase 1: Scans for APIs only (internal + external) - FAST
Phase 2: Scans for services using selected APIs - TARGETED

Usage:
    python scan-codebase.py [repo_root] [--phase 1|2] [--selected-apis "api1,api2"]
    
If no repo_root provided, uses current working directory.
Phase 1 is default if not specified.
"""

import os
import re
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional


# Supported Temporal languages and their file extensions
TEMPORAL_LANGUAGES = {
    'typescript': ['.ts', '.tsx'],
    'javascript': ['.js', '.jsx'],
    'python': ['.py'],
    'go': ['.go'],
    'java': ['.java'],
    'csharp': ['.cs'],
    'php': ['.php'],
    'ruby': ['.rb']
}

# Patterns to exclude from scanning
EXCLUDE_PATTERNS = [
    'node_modules',
    '.git',
    '.venv',
    'venv',
    '__pycache__',
    '.pytest_cache',
    'dist',
    'build',
    '.next',
    '.cursor',
    'cursor-skills-temporal',  # Exclude the system itself
    '.env',
    '.env.local'
]


def should_exclude_path(file_path: str) -> bool:
    """Check if file path should be excluded from scanning"""
    path_parts = Path(file_path).parts
    for exclude in EXCLUDE_PATTERNS:
        if exclude in path_parts:
            return True
    return False


def detect_language(file_path: str) -> Optional[str]:
    """Detect programming language from file extension"""
    ext = Path(file_path).suffix.lower()
    for lang, extensions in TEMPORAL_LANGUAGES.items():
        if ext in extensions:
            return lang
    return None


def find_code_files(repo_root: str) -> List[Dict[str, str]]:
    """Find all code files in repository"""
    code_files = []
    repo_path = Path(repo_root)
    
    for file_path in repo_path.rglob('*'):
        if not file_path.is_file():
            continue
            
        file_str = str(file_path)
        if should_exclude_path(file_str):
            continue
            
        language = detect_language(file_str)
        if language:
            code_files.append({
                'path': str(file_path.relative_to(repo_path)),
                'full_path': file_str,
                'language': language
            })
    
    return code_files


def detect_temporal_imports(content: str, language: str) -> bool:
    """Detect Temporal SDK imports"""
    temporal_patterns = {
        'typescript': [
            r'from\s+["\']@temporalio',
            r'import\s+.*from\s+["\']@temporalio',
            r'require\(["\']@temporalio'
        ],
        'javascript': [
            r'from\s+["\']@temporalio',
            r'import\s+.*from\s+["\']@temporalio',
            r'require\(["\']@temporalio'
        ],
        'python': [
            r'from\s+temporalio',
            r'import\s+temporalio',
            r'from\s+temporal'
        ],
        'go': [
            r'"go\.temporal\.io',
            r'github\.com/temporalio'
        ],
        'java': [
            r'import\s+io\.temporal',
            r'import\s+com\.uber\.cadence'
        ],
        'csharp': [
            r'using\s+Temporalio',
            r'using\s+Temporal\.'
        ],
        'php': [
            r'use\s+Temporal\\',
            r'require.*temporal'
        ],
        'ruby': [
            r'require\s+["\']temporal',
            r'require_relative.*temporal'
        ]
    }
    
    patterns = temporal_patterns.get(language, [])
    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def detect_workflow_definitions(content: str, language: str) -> List[str]:
    """Detect workflow definitions"""
    workflows = []
    
    patterns = {
        'typescript': r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*:\s*Promise<[^>]+>',
        'javascript': r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
        'python': r'@?workflow\.(?:def|run)\s+def\s+(\w+)',
        'go': r'func\s+(\w+)\s*\([^)]*\)\s*\([^)]+\)',
        'java': r'(?:public\s+)?(?:static\s+)?\w+\s+(\w+)\s*\([^)]*\)',
        'csharp': r'(?:public\s+)?(?:static\s+)?\w+\s+(\w+)\s*\([^)]*\)',
        'php': r'function\s+(\w+)\s*\([^)]*\)',
        'ruby': r'def\s+(\w+)'
    }
    
    pattern = patterns.get(language, r'function\s+(\w+)')
    matches = re.finditer(pattern, content, re.MULTILINE)
    
    # Check if function is decorated with @workflow or similar
    workflow_keywords = ['workflow', 'Workflow', '@workflow', 'defineWorkflow']
    for match in matches:
        func_name = match.group(1)
        # Check context around function for workflow indicators
        start = max(0, match.start() - 200)
        context = content[start:match.end()]
        if any(keyword in context for keyword in workflow_keywords):
            workflows.append(func_name)
    
    return workflows


def detect_activity_definitions(content: str, language: str) -> List[str]:
    """Detect activity definitions"""
    activities = []
    
    activity_keywords = ['@activity', 'activity', 'Activity', 'defineActivity']
    patterns = {
        'typescript': r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
        'javascript': r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
        'python': r'@?activity\.(?:def|run)\s+def\s+(\w+)',
        'go': r'func\s+(\w+)\s*\([^)]*\)',
        'java': r'(?:public\s+)?(?:static\s+)?\w+\s+(\w+)\s*\([^)]*\)',
        'csharp': r'(?:public\s+)?(?:static\s+)?\w+\s+(\w+)\s*\([^)]*\)',
        'php': r'function\s+(\w+)\s*\([^)]*\)',
        'ruby': r'def\s+(\w+)'
    }
    
    pattern = patterns.get(language, r'function\s+(\w+)')
    matches = re.finditer(pattern, content, re.MULTILINE)
    
    for match in matches:
        func_name = match.group(1)
        start = max(0, match.start() - 200)
        context = content[start:match.end()]
        if any(keyword in context for keyword in activity_keywords):
            activities.append(func_name)
    
    return activities


def detect_worker_setups(content: str, language: str) -> bool:
    """Detect worker setup/configuration"""
    worker_keywords = [
        'Worker.create',
        'new Worker',
        'WorkerOptions',
        'Worker.createOptions',
        'worker.run',
        'worker.start',
        'createWorker',
        'defineWorker'
    ]
    
    return any(keyword in content for keyword in worker_keywords)


def detect_external_apis(content: str, language: str, file_path: str) -> List[Dict]:
    """Detect external API calls and their usage"""
    apis_found = []
    
    # API detection patterns by import/library
    api_import_patterns = {
        'typescript': {
            r'import.*openai|from\s+["\']openai': 'OpenAI API',
            r'import.*@anthropic-ai/sdk|from\s+["\']@anthropic': 'Claude API (Anthropic)',
            r'import.*@google/generative-ai|from\s+["\']@google': 'Gemini API (Google)',
            r'import.*axios|from\s+["\']axios': 'Axios HTTP Client',
            r'import.*googleapis|from\s+["\']googleapis': 'Google APIs',
            r'import.*stripe|from\s+["\']stripe': 'Stripe API',
            r'import.*twilio|from\s+["\']twilio': 'Twilio API',
            r'import.*aws-sdk|from\s+["\']aws-sdk': 'AWS SDK',
            r'import.*@azure|from\s+["\']@azure': 'Azure SDK',
            r'import.*sendgrid|from\s+["\']@sendgrid': 'SendGrid API'
        },
        'javascript': {
            r'require\(["\']openai|import.*openai': 'OpenAI API',
            r'require\(["\']@anthropic|import.*@anthropic': 'Claude API (Anthropic)',
            r'require\(["\']@google/generative-ai': 'Gemini API (Google)',
            r'require\(["\']axios|import.*axios': 'Axios HTTP Client',
            r'require\(["\']googleapis': 'Google APIs',
            r'require\(["\']stripe': 'Stripe API',
            r'require\(["\']twilio': 'Twilio API',
            r'require\(["\']aws-sdk': 'AWS SDK',
            r'require\(["\']@azure': 'Azure SDK'
        },
        'python': {
            r'import\s+openai|from\s+openai': 'OpenAI API',
            r'import\s+anthropic|from\s+anthropic': 'Claude API (Anthropic)',
            r'import\s+google\.generativeai|from\s+google\.generativeai': 'Gemini API (Google)',
            r'import\s+requests|from\s+requests': 'Requests HTTP Library',
            r'import\s+httpx|from\s+httpx': 'HTTPX Library',
            r'import\s+urllib|from\s+urllib': 'urllib HTTP Library',
            r'from\s+google\.oauth2|import\s+google\.auth': 'Google APIs',
            r'import\s+stripe|from\s+stripe': 'Stripe API',
            r'import\s+twilio|from\s+twilio': 'Twilio API',
            r'import\s+boto3|from\s+boto3': 'AWS SDK (boto3)',
            r'from\s+azure|import\s+azure': 'Azure SDK',
            r'import\s+sendgrid|from\s+sendgrid': 'SendGrid API',
            r'import\s+pymongo|from\s+pymongo': 'MongoDB',
            r'import\s+psycopg2|from\s+psycopg2': 'PostgreSQL',
            r'import\s+mysql\.connector|from\s+mysql': 'MySQL',
            r'import\s+redis|from\s+redis': 'Redis'
        },
        'go': {
            r'"github\.com/openai': 'OpenAI API',
            r'"github\.com/anthropic': 'Claude API (Anthropic)',
            r'"cloud\.google\.com/go/ai': 'Gemini API (Google)',
            r'"net/http"': 'HTTP Client',
            r'"github\.com/stripe/stripe-go': 'Stripe API',
            r'"github\.com/twilio': 'Twilio API',
            r'"github\.com/aws/aws-sdk-go': 'AWS SDK'
        },
        'java': {
            r'import\s+com\.openai': 'OpenAI API',
            r'import\s+com\.anthropic': 'Claude API (Anthropic)',
            r'import\s+com\.google\.cloud\.aiplatform': 'Gemini API (Google)',
            r'import\s+org\.apache\.http|import\s+java\.net\.http': 'HTTP Client',
            r'import\s+com\.stripe': 'Stripe API',
            r'import\s+com\.twilio': 'Twilio API',
            r'import\s+com\.amazonaws': 'AWS SDK'
        },
        'csharp': {
            r'using\s+OpenAI': 'OpenAI API',
            r'using\s+Anthropic': 'Claude API (Anthropic)',
            r'using\s+Google\.Cloud\.AIPlatform': 'Gemini API (Google)',
            r'using\s+System\.Net\.Http': 'HTTP Client',
            r'using\s+Stripe': 'Stripe API',
            r'using\s+Twilio': 'Twilio API',
            r'using\s+Amazon\.': 'AWS SDK',
            r'using\s+Microsoft\.Azure': 'Azure SDK'
        },
        'php': {
            r'use\s+OpenAI': 'OpenAI API',
            r'use\s+Anthropic': 'Claude API (Anthropic)',
            r'use\s+Google\\Cloud\\': 'Google APIs',
            r'use\s+GuzzleHttp': 'Guzzle HTTP Client',
            r'use\s+Stripe\\': 'Stripe API',
            r'use\s+Twilio\\': 'Twilio API'
        },
        'ruby': {
            r'require\s+["\']openai': 'OpenAI API',
            r'require\s+["\']anthropic': 'Claude API (Anthropic)',
            r'require\s+["\']google': 'Google APIs',
            r'require\s+["\']faraday|require\s+["\']httparty': 'HTTP Client',
            r'require\s+["\']stripe': 'Stripe API',
            r'require\s+["\']twilio': 'Twilio API'
        }
    }
    
    # HTTP call patterns (direct URL calls)
    http_call_patterns = {
        'typescript': r'(?:fetch|axios\.(?:get|post|put|delete))\s*\(["\']([^"\']+)',
        'javascript': r'(?:fetch|axios\.(?:get|post|put|delete))\s*\(["\']([^"\']+)',
        'python': r'(?:requests\.(?:get|post|put|delete)|urllib\.request\.urlopen|httpx\.(?:get|post))\s*\(["\']([^"\']+)',
        'go': r'http\.(?:Get|Post)\s*\(["\']([^"\']+)',
        'java': r'(?:HttpClient|RestTemplate).*?\.(?:get|post)\s*\(["\']([^"\']+)',
        'csharp': r'HttpClient.*?\.(?:GetAsync|PostAsync)\s*\(["\']([^"\']+)',
        'php': r'(?:curl_setopt|file_get_contents|Guzzle.*?->(?:get|post))\s*\(["\']([^"\']+)',
        'ruby': r'(?:Net::HTTP|HTTParty|Faraday).*?\.(?:get|post)\s*\(["\']([^"\']+)'
    }
    
    # Detect APIs by imports
    import_patterns = api_import_patterns.get(language, {})
    for pattern, api_name in import_patterns.items():
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            apis_found.append({
                'api_name': api_name,
                'detection_method': 'import',
                'file': file_path,
                'line': line_num,
                'context': match.group(0)
            })
    
    # Detect APIs by HTTP calls
    http_pattern = http_call_patterns.get(language, '')
    if http_pattern:
        matches = re.finditer(http_pattern, content, re.IGNORECASE)
        for match in matches:
            url = match.group(1) if match.lastindex >= 1 else 'unknown'
            line_num = content[:match.start()].count('\n') + 1
            
            # Extract API name from URL
            api_name = 'Unknown HTTP API'
            if 'openai.com' in url or 'api.openai' in url:
                api_name = 'OpenAI API'
            elif 'anthropic.com' in url or 'claude' in url:
                api_name = 'Claude API (Anthropic)'
            elif 'generativelanguage.googleapis.com' in url or 'gemini' in url:
                api_name = 'Gemini API (Google)'
            elif 'stripe.com' in url:
                api_name = 'Stripe API'
            elif 'twilio.com' in url:
                api_name = 'Twilio API'
            elif 'amazonaws.com' in url:
                api_name = 'AWS API'
            elif 'azure.com' in url or 'windows.net' in url:
                api_name = 'Azure API'
            elif 'googleapis.com' in url or 'google.com/api' in url:
                api_name = 'Google APIs'
            else:
                api_name = f'HTTP API ({url[:50]}...)'
            
            apis_found.append({
                'api_name': api_name,
                'detection_method': 'http_call',
                'file': file_path,
                'line': line_num,
                'url': url,
                'context': match.group(0)
            })
    
    return apis_found


def detect_internal_apis(content: str, language: str, file_path: str) -> List[Dict]:
    """
    Detect internal APIs with comprehensive metadata extraction.
    Captures: annotations, path, method, content-type, parameters, function name, 
    line number, additional annotations, category, and priority.
    """
    internal_apis = []
    
    # Language-specific comprehensive detection
    if language == 'java':
        internal_apis.extend(detect_java_apis(content, file_path))
    elif language in ['typescript', 'javascript']:
        internal_apis.extend(detect_typescript_apis(content, file_path, language))
    elif language == 'python':
        internal_apis.extend(detect_python_apis(content, file_path))
    elif language == 'go':
        internal_apis.extend(detect_go_apis(content, file_path))
    elif language == 'csharp':
        internal_apis.extend(detect_csharp_apis(content, file_path))
    elif language == 'php':
        internal_apis.extend(detect_php_apis(content, file_path))
    elif language == 'ruby':
        internal_apis.extend(detect_ruby_apis(content, file_path))
    
    return internal_apis


def detect_java_apis(content: str, file_path: str) -> List[Dict]:
    """Comprehensive Java API detection (Spring, JAX-RS)"""
    apis = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Spring annotations
        spring_annotations = ['@GetMapping', '@PostMapping', '@PutMapping', '@DeleteMapping', '@PatchMapping', '@RequestMapping']
        jaxrs_annotations = ['@GET', '@POST', '@PUT', '@DELETE', '@PATCH']
        
        for annotation in spring_annotations + jaxrs_annotations:
            if annotation in line:
                # Extract comprehensive metadata
                api_data = {
                    'type': 'http_route',
                    'framework': 'Spring' if annotation.endswith('Mapping') else 'JAX-RS',
                    'file': file_path,
                    'line': line_num,
                    'annotations': [],
                    'path': '',
                    'method': '',
                    'content_type': {'consumes': [], 'produces': []},
                    'parameters': [],
                    'function_name': '',
                    'additional_annotations': [],
                    'category': 'REST',
                    'priority': 'MEDIUM'
                }
                
                # Extract HTTP method
                if annotation == '@GetMapping' or annotation == '@GET':
                    api_data['method'] = 'GET'
                elif annotation == '@PostMapping' or annotation == '@POST':
                    api_data['method'] = 'POST'
                elif annotation == '@PutMapping' or annotation == '@PUT':
                    api_data['method'] = 'PUT'
                elif annotation == '@DeleteMapping' or annotation == '@DELETE':
                    api_data['method'] = 'DELETE'
                elif annotation == '@PatchMapping' or annotation == '@PATCH':
                    api_data['method'] = 'PATCH'
                elif annotation == '@RequestMapping':
                    # Extract method from annotation parameters
                    method_match = re.search(r'method\s*=\s*RequestMethod\.(\w+)', line)
                    api_data['method'] = method_match.group(1) if method_match else 'UNKNOWN'
                
                api_data['annotations'].append(annotation)
                
                # Extract path/value
                path_match = re.search(r'(?:value|path)\s*=\s*"([^"]+)"', line)
                if not path_match:
                    path_match = re.search(r'@\w+Mapping\s*\(\s*"([^"]+)"', line)
                if path_match:
                    api_data['path'] = path_match.group(1)
                
                # Extract consumes
                consumes_match = re.search(r'consumes\s*=\s*[{"]([^"}]+)["}]', line)
                if consumes_match:
                    api_data['content_type']['consumes'] = [c.strip().strip('"') for c in consumes_match.group(1).split(',')]
                
                # Extract produces
                produces_match = re.search(r'produces\s*=\s*[{"]([^"}]+)["}]', line)
                if produces_match:
                    api_data['content_type']['produces'] = [p.strip().strip('"') for p in produces_match.group(1).split(',')]
                
                # Look ahead for function name and parameters
                for j in range(i + 1, min(i + 10, len(lines))):
                    func_line = lines[j]
                    
                    # Extract function name
                    func_match = re.search(r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*\(', func_line)
                    if func_match:
                        api_data['function_name'] = func_match.group(1)
                        
                        # Extract parameters (@RequestBody, @PathVariable, @RequestParam, etc.)
                        params = re.findall(r'(@RequestBody|@PathVariable|@RequestParam|@RequestHeader)\s*(?:\([^)]*\))?\s*\w+\s+(\w+)', func_line)
                        api_data['parameters'] = [{'annotation': p[0], 'name': p[1]} for p in params]
                        break
                
                # Look backwards for additional annotations
                for j in range(max(0, i - 5), i):
                    prev_line = lines[j]
                    additional_annotations = re.findall(r'@(CrossOrigin|Retryable|Transactional|Async|Cacheable|Secured|PreAuthorize|Valid|Validated)', prev_line)
                    api_data['additional_annotations'].extend(additional_annotations)
                
                # Calculate priority based on complexity
                complexity_score = 0
                complexity_score += len(api_data['parameters']) * 2
                complexity_score += len(api_data['additional_annotations'])
                complexity_score += 1 if api_data['content_type']['consumes'] else 0
                complexity_score += 1 if api_data['content_type']['produces'] else 0
                
                if complexity_score >= 5:
                    api_data['priority'] = 'HIGH'
                elif complexity_score >= 2:
                    api_data['priority'] = 'MEDIUM'
                else:
                    api_data['priority'] = 'LOW'
                
                # Create endpoint identifier
                api_data['endpoint'] = f"{api_data['method']} {api_data['path']}" if api_data['path'] else f"{api_data['method']} (path not found)"
                
                apis.append(api_data)
    
    return apis


def detect_typescript_apis(content: str, file_path: str, language: str) -> List[Dict]:
    """Comprehensive TypeScript/JavaScript API detection (Express, NestJS, Fastify)"""
    apis = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # NestJS decorators
        nestjs_decorators = ['@Get', '@Post', '@Put', '@Delete', '@Patch']
        for decorator in nestjs_decorators:
            if decorator in line:
                api_data = {
                    'type': 'http_route',
                    'framework': 'NestJS',
                    'file': file_path,
                    'line': line_num,
                    'annotations': [decorator],
                    'path': '',
                    'method': decorator[1:].upper(),
                    'content_type': {'consumes': [], 'produces': []},
                    'parameters': [],
                    'function_name': '',
                    'additional_annotations': [],
                    'category': 'REST',
                    'priority': 'MEDIUM'
                }
                
                # Extract path
                path_match = re.search(r'@\w+\s*\(\s*["\']([^"\']+)', line)
                if path_match:
                    api_data['path'] = path_match.group(1)
                
                # Look ahead for function name and parameters
                for j in range(i + 1, min(i + 10, len(lines))):
                    func_line = lines[j]
                    func_match = re.search(r'(?:async\s+)?(\w+)\s*\(', func_line)
                    if func_match:
                        api_data['function_name'] = func_match.group(1)
                        
                        # Extract NestJS parameters
                        params = re.findall(r'@(Body|Param|Query|Headers?)\s*\([^)]*\)\s*(\w+)', func_line)
                        api_data['parameters'] = [{'annotation': '@' + p[0], 'name': p[1]} for p in params]
                        break
                
                # Look for additional decorators
                for j in range(max(0, i - 5), i):
                    prev_line = lines[j]
                    additional = re.findall(r'@(UseGuards|UsePipes|UseInterceptors|UseFilters|HttpCode|Header)', prev_line)
                    api_data['additional_annotations'].extend(additional)
                
                # Calculate priority
                complexity = len(api_data['parameters']) * 2 + len(api_data['additional_annotations'])
                api_data['priority'] = 'HIGH' if complexity >= 5 else 'MEDIUM' if complexity >= 2 else 'LOW'
                
                api_data['endpoint'] = f"{api_data['method']} {api_data['path']}" if api_data['path'] else f"{api_data['method']} (path not found)"
                apis.append(api_data)
        
        # Express.js route patterns
        express_match = re.search(r'(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)', line)
        if express_match:
            method = express_match.group(1).upper()
            path = express_match.group(2)
            
            api_data = {
                'type': 'http_route',
                'framework': 'Express',
                'file': file_path,
                'line': line_num,
                'annotations': ['express.' + express_match.group(1)],
                'path': path,
                'method': method,
                'content_type': {'consumes': [], 'produces': []},
                'parameters': [],
                'function_name': '',
                'additional_annotations': [],
                'category': 'REST',
                'priority': 'MEDIUM',
                'endpoint': f'{method} {path}'
            }
            
            # Look for route parameters
            params = re.findall(r':(\w+)', path)
            api_data['parameters'] = [{'annotation': 'path_param', 'name': p} for p in params]
            
            apis.append(api_data)
    
    return apis


def detect_python_apis(content: str, file_path: str) -> List[Dict]:
    """Comprehensive Python API detection (Flask, FastAPI, Django)"""
    apis = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # FastAPI decorators
        fastapi_match = re.search(r'@(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)', line)
        if fastapi_match:
            method = fastapi_match.group(1).upper()
            path = fastapi_match.group(2)
            
            api_data = {
                'type': 'http_route',
                'framework': 'FastAPI',
                'file': file_path,
                'line': line_num,
                'annotations': ['@app.' + fastapi_match.group(1)],
                'path': path,
                'method': method,
                'content_type': {'consumes': [], 'produces': []},
                'parameters': [],
                'function_name': '',
                'additional_annotations': [],
                'category': 'REST',
                'priority': 'MEDIUM'
            }
            
            # Extract response_model, status_code, etc.
            if 'response_model' in line:
                api_data['additional_annotations'].append('response_model')
            if 'status_code' in line:
                api_data['additional_annotations'].append('status_code')
            
            # Look ahead for function definition
            for j in range(i + 1, min(i + 5, len(lines))):
                func_line = lines[j]
                func_match = re.search(r'(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)', func_line)
                if func_match:
                    api_data['function_name'] = func_match.group(1)
                    params_str = func_match.group(2)
                    
                    # Extract parameters (Path, Query, Body, Header)
                    params = re.findall(r'(\w+):\s*(?:Path|Query|Body|Header|Form|File)', params_str)
                    api_data['parameters'] = [{'annotation': 'type_hint', 'name': p} for p in params]
                    break
            
            complexity = len(api_data['parameters']) * 2 + len(api_data['additional_annotations'])
            api_data['priority'] = 'HIGH' if complexity >= 5 else 'MEDIUM' if complexity >= 2 else 'LOW'
            api_data['endpoint'] = f'{method} {path}'
            
            apis.append(api_data)
        
        # Flask decorators
        flask_match = re.search(r'@app\.route\s*\(\s*["\']([^"\']+)["\'].*methods\s*=\s*\[([^\]]+)\]', line)
        if flask_match:
            path = flask_match.group(1)
            methods = [m.strip().strip('"\'') for m in flask_match.group(2).split(',')]
            
            for method in methods:
                api_data = {
                    'type': 'http_route',
                    'framework': 'Flask',
                    'file': file_path,
                    'line': line_num,
                    'annotations': ['@app.route'],
                    'path': path,
                    'method': method.upper(),
                    'content_type': {'consumes': [], 'produces': []},
                    'parameters': [],
                    'function_name': '',
                    'additional_annotations': [],
                    'category': 'REST',
                    'priority': 'MEDIUM',
                    'endpoint': f'{method.upper()} {path}'
                }
                
                # Extract function name
                for j in range(i + 1, min(i + 3, len(lines))):
                    func_match = re.search(r'def\s+(\w+)\s*\(', lines[j])
                    if func_match:
                        api_data['function_name'] = func_match.group(1)
                        break
                
                apis.append(api_data)
    
    return apis


def detect_go_apis(content: str, file_path: str) -> List[Dict]:
    """Comprehensive Go API detection (Gin, Echo, Chi, Gorilla)"""
    apis = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Gin/Echo/Chi patterns
        go_match = re.search(r'(?:router|r|e|mux)\.(GET|POST|PUT|DELETE|PATCH|Handle)\s*\(\s*"([^"]+)"', line)
        if go_match:
            method = go_match.group(1)
            path = go_match.group(2)
            
            api_data = {
                'type': 'http_route',
                'framework': 'Go (Gin/Echo/Chi)',
                'file': file_path,
                'line': line_num,
                'annotations': [],
                'path': path,
                'method': method if method != 'Handle' else 'UNKNOWN',
                'content_type': {'consumes': [], 'produces': []},
                'parameters': [],
                'function_name': '',
                'additional_annotations': [],
                'category': 'REST',
                'priority': 'MEDIUM',
                'endpoint': f'{method} {path}'
            }
            
            # Extract handler function name
            handler_match = re.search(r'(?:router|r|e|mux)\.\w+\s*\([^,]+,\s*(\w+)', line)
            if handler_match:
                api_data['function_name'] = handler_match.group(1)
            
            # Extract path parameters
            params = re.findall(r':(\w+)', path)
            api_data['parameters'] = [{'annotation': 'path_param', 'name': p} for p in params]
            
            apis.append(api_data)
    
    return apis


def detect_csharp_apis(content: str, file_path: str) -> List[Dict]:
    """Comprehensive C# API detection (ASP.NET)"""
    apis = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # ASP.NET attributes
        aspnet_match = re.search(r'\[(HttpGet|HttpPost|HttpPut|HttpDelete|HttpPatch)\s*\(\s*"([^"]+)"', line)
        if aspnet_match:
            method = aspnet_match.group(1).replace('Http', '').upper()
            path = aspnet_match.group(2)
            
            api_data = {
                'type': 'http_route',
                'framework': 'ASP.NET',
                'file': file_path,
                'line': line_num,
                'annotations': [aspnet_match.group(1)],
                'path': path,
                'method': method,
                'content_type': {'consumes': [], 'produces': []},
                'parameters': [],
                'function_name': '',
                'additional_annotations': [],
                'category': 'REST',
                'priority': 'MEDIUM',
                'endpoint': f'{method} {path}'
            }
            
            # Look ahead for function
            for j in range(i + 1, min(i + 5, len(lines))):
                func_match = re.search(r'(?:public|private)?\s+\w+\s+(\w+)\s*\(', lines[j])
                if func_match:
                    api_data['function_name'] = func_match.group(1)
                    break
            
            apis.append(api_data)
    
    return apis


def detect_php_apis(content: str, file_path: str) -> List[Dict]:
    """Comprehensive PHP API detection (Laravel, Symfony)"""
    apis = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Laravel routes
        laravel_match = re.search(r'Route::(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)', line)
        if laravel_match:
            method = laravel_match.group(1).upper()
            path = laravel_match.group(2)
            
            api_data = {
                'type': 'http_route',
                'framework': 'Laravel',
                'file': file_path,
                'line': line_num,
                'annotations': ['Route::' + laravel_match.group(1)],
                'path': path,
                'method': method,
                'content_type': {'consumes': [], 'produces': []},
                'parameters': [],
                'function_name': '',
                'additional_annotations': [],
                'category': 'REST',
                'priority': 'MEDIUM',
                'endpoint': f'{method} {path}'
            }
            
            apis.append(api_data)
    
    return apis


def detect_ruby_apis(content: str, file_path: str) -> List[Dict]:
    """Comprehensive Ruby API detection (Rails, Sinatra)"""
    apis = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Rails/Sinatra routes
        ruby_match = re.search(r'(get|post|put|delete|patch)\s+["\']([^"\']+)', line)
        if ruby_match:
            method = ruby_match.group(1).upper()
            path = ruby_match.group(2)
            
            api_data = {
                'type': 'http_route',
                'framework': 'Rails/Sinatra',
                'file': file_path,
                'line': line_num,
                'annotations': [ruby_match.group(1)],
                'path': path,
                'method': method,
                'content_type': {'consumes': [], 'produces': []},
                'parameters': [],
                'function_name': '',
                'additional_annotations': [],
                'category': 'REST',
                'priority': 'MEDIUM',
                'endpoint': f'{method} {path}'
            }
            
            apis.append(api_data)
    
    return apis


def detect_patterns(content: str, language: str) -> Dict[str, List[Dict]]:
    """Detect code patterns suitable for Temporal"""
    patterns = {
        'long_running': [],
        'multi_step': [],
        'external_api': [],
        'background_jobs': [],
        'scheduled_tasks': []
    }
    
    # Long-running operations (functions with async/await, timeouts, delays)
    long_running_patterns = {
        'typescript': r'(?:async\s+)?function\s+(\w+).*?setTimeout|setInterval|sleep|delay|wait',
        'javascript': r'(?:async\s+)?function\s+(\w+).*?setTimeout|setInterval|sleep|delay|wait',
        'python': r'def\s+(\w+).*?time\.sleep|asyncio\.sleep|await.*sleep',
        'go': r'func\s+(\w+).*?time\.Sleep|time\.After',
        'java': r'(?:public\s+)?(?:static\s+)?\w+\s+(\w+).*?Thread\.sleep|TimeUnit\.',
        'csharp': r'(?:public\s+)?(?:static\s+)?\w+\s+(\w+).*?Thread\.Sleep|Task\.Delay',
        'php': r'function\s+(\w+).*?sleep|usleep',
        'ruby': r'def\s+(\w+).*?sleep|wait'
    }
    
    # Multi-step processes (sequential function calls)
    multi_step_patterns = {
        'typescript': r'(?:async\s+)?function\s+(\w+).*?await\s+\w+\(.*?\)\s*;.*?await\s+\w+\(',
        'javascript': r'(?:async\s+)?function\s+(\w+).*?await\s+\w+\(.*?\)\s*;.*?await\s+\w+\(',
        'python': r'def\s+(\w+).*?\w+\(.*?\)\s*\n.*?\w+\(.*?\)\s*\n.*?\w+\(',
        'go': r'func\s+(\w+).*?\w+\(.*?\)\s*\n.*?\w+\(.*?\)\s*\n.*?\w+\(',
        'java': r'(?:public\s+)?(?:static\s+)?\w+\s+(\w+).*?\w+\(.*?\)\s*;.*?\w+\(.*?\)\s*;.*?\w+\(',
        'csharp': r'(?:public\s+)?(?:static\s+)?\w+\s+(\w+).*?\w+\(.*?\)\s*;.*?\w+\(.*?\)\s*;.*?\w+\(',
        'php': r'function\s+(\w+).*?\w+\(.*?\)\s*;.*?\w+\(.*?\)\s*;.*?\w+\(',
        'ruby': r'def\s+(\w+).*?\w+\(.*?\)\s*\n.*?\w+\(.*?\)\s*\n.*?\w+\('
    }
    
    # External API calls
    api_patterns = {
        'typescript': r'(?:fetch|axios|http\.(?:get|post|put|delete)|request)',
        'javascript': r'(?:fetch|axios|http\.(?:get|post|put|delete)|request)',
        'python': r'(?:requests\.(?:get|post|put|delete)|urllib|httpx)',
        'go': r'(?:http\.(?:Get|Post|Put|Delete)|net/http)',
        'java': r'(?:HttpClient|RestTemplate|OkHttp)',
        'csharp': r'(?:HttpClient|WebRequest|RestClient)',
        'php': r'(?:curl_|file_get_contents|guzzle)',
        'ruby': r'(?:Net::HTTP|HTTParty|Faraday)'
    }
    
    # Background jobs
    job_patterns = {
        'typescript': r'(?:queue|job|worker|background|async|setTimeout|setInterval)',
        'javascript': r'(?:queue|job|worker|background|async|setTimeout|setInterval)',
        'python': r'(?:celery|rq|queue|job|worker|background)',
        'go': r'(?:queue|job|worker|goroutine|go\s+func)',
        'java': r'(?:ExecutorService|ThreadPool|@Async|@Scheduled)',
        'csharp': r'(?:Task\.Run|BackgroundService|IHostedService)',
        'php': r'(?:queue|job|worker|background)',
        'ruby': r'(?:Sidekiq|Resque|DelayedJob|queue)'
    }
    
    # Scheduled tasks
    scheduled_patterns = {
        'typescript': r'(?:cron|schedule|setInterval|@Scheduled)',
        'javascript': r'(?:cron|schedule|setInterval)',
        'python': r'(?:cron|crontab|schedule|@scheduled|APScheduler)',
        'go': r'(?:cron|schedule|ticker)',
        'java': r'(?:@Scheduled|cron|Timer)',
        'csharp': r'(?:@Scheduled|Timer|CronExpression)',
        'php': r'(?:cron|schedule)',
        'ruby': r'(?:cron|schedule|whenever)'
    }
    
    # Detect patterns
    for match in re.finditer(long_running_patterns.get(language, r'function\s+(\w+)'), content, re.MULTILINE | re.DOTALL):
        patterns['long_running'].append({
            'function': match.group(1),
            'line': content[:match.start()].count('\n') + 1
        })
    
    for match in re.finditer(multi_step_patterns.get(language, r'function\s+(\w+)'), content, re.MULTILINE | re.DOTALL):
        patterns['multi_step'].append({
            'function': match.group(1),
            'line': content[:match.start()].count('\n') + 1
        })
    
    if re.search(api_patterns.get(language, ''), content, re.IGNORECASE):
        # Find function containing API call
        for match in re.finditer(r'function\s+(\w+)', content, re.MULTILINE):
            patterns['external_api'].append({
                'function': match.group(1),
                'line': content[:match.start()].count('\n') + 1
            })
    
    if re.search(job_patterns.get(language, ''), content, re.IGNORECASE):
        for match in re.finditer(r'function\s+(\w+)', content, re.MULTILINE):
            patterns['background_jobs'].append({
                'function': match.group(1),
                'line': content[:match.start()].count('\n') + 1
            })
    
    if re.search(scheduled_patterns.get(language, ''), content, re.IGNORECASE):
        for match in re.finditer(r'function\s+(\w+)', content, re.MULTILINE):
            patterns['scheduled_tasks'].append({
                'function': match.group(1),
                'line': content[:match.start()].count('\n') + 1
            })
    
    return patterns


def aggregate_internal_apis(internal_apis: List[Dict]) -> List[Dict]:
    """Deduplicate and aggregate internal APIs with comprehensive metadata"""
    api_map = {}
    
    for api in internal_apis:
        endpoint = api.get('endpoint', 'unknown')
        
        if endpoint not in api_map:
            api_map[endpoint] = {
                'endpoint': endpoint,
                'type': api.get('type', 'unknown'),
                'method': api.get('method', 'UNKNOWN'),
                'path': api.get('path', ''),
                'framework': api.get('framework', 'detected'),
                'annotations': api.get('annotations', []),
                'content_type': api.get('content_type', {'consumes': [], 'produces': []}),
                'parameters': api.get('parameters', []),
                'function_name': api.get('function_name', ''),
                'additional_annotations': api.get('additional_annotations', []),
                'category': api.get('category', 'REST'),
                'priority': api.get('priority', 'MEDIUM'),
                'occurrences': 0,
                'files': [],
                'locations': []
            }
        
        api_map[endpoint]['occurrences'] += 1
        if api.get('file') not in api_map[endpoint]['files']:
            api_map[endpoint]['files'].append(api.get('file'))
        api_map[endpoint]['locations'].append({
            'file': api.get('file'),
            'line': api.get('line'),
            'function': api.get('function_name', '')
        })
    
    # Sort by priority (HIGH, MEDIUM, LOW) then by occurrences
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    sorted_apis = sorted(
        api_map.values(),
        key=lambda x: (priority_order.get(x['priority'], 3), -x['occurrences'])
    )
    
    return sorted_apis


def aggregate_external_apis(external_apis: List[Dict]) -> List[Dict]:
    """Deduplicate and aggregate external APIs"""
    api_map = {}
    
    for api in external_apis:
        api_name = api.get('api_name', 'Unknown API')
        
        if api_name not in api_map:
            api_map[api_name] = {
                'api_name': api_name,
                'call_count': 0,
                'file_count': 0,
                'files': set(),
                'usages': []
            }
        
        api_map[api_name]['call_count'] += 1
        api_map[api_name]['files'].add(api.get('file'))
        api_map[api_name]['usages'].append({
            'file': api.get('file'),
            'line': api.get('line'),
            'method': api.get('detection_method', 'unknown'),
            'context': api.get('context', '')[:100]
        })
    
    # Convert sets to lists
    for api_name in api_map:
        api_map[api_name]['files'] = list(api_map[api_name]['files'])
        api_map[api_name]['file_count'] = len(api_map[api_name]['files'])
    
    # Sort by call count (descending)
    sorted_apis = sorted(api_map.values(), key=lambda x: x['call_count'], reverse=True)
    
    # Add priority levels
    for api in sorted_apis:
        if api['call_count'] >= 10:
            api['priority'] = 'HIGH'
        elif api['call_count'] >= 3:
            api['priority'] = 'MEDIUM'
        else:
            api['priority'] = 'LOW'
    
    return sorted_apis


def find_services_using_api(code_files: List[Dict], api_id: str, api_type: str, repo_root: str) -> List[Dict]:
    """Find all services/functions using specified API"""
    services = []
    
    # Parse api_id to extract search patterns
    if api_type == 'internal':
        # api_id format: "METHOD /path" or "GRAPHQL resolver" or "GRPC method"
        search_pattern = api_id.split(' ', 1)[-1]  # Get path/name after method
    else:
        # External API - search for API name
        search_pattern = api_id
    
    for file_info in code_files:
        try:
            with open(file_info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Search for API usage in file
            if search_pattern.lower() in content.lower():
                # Find functions/methods in this file
                language = file_info['language']
                functions = find_functions_in_file(content, language)
                
                for func in functions:
                    # Check if this function uses the API
                    func_content = content[func['start']:func['end']]
                    if search_pattern.lower() in func_content.lower():
                        services.append({
                            'service_name': func['name'],
                            'file': file_info['path'],
                            'line': func['line'],
                            'language': language,
                            'api_usage_count': func_content.lower().count(search_pattern.lower())
                        })
        
        except Exception as e:
            continue
    
    return services


def find_functions_in_file(content: str, language: str) -> List[Dict]:
    """Find all function definitions in a file"""
    functions = []
    
    patterns = {
        'typescript': r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',
        'javascript': r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',
        'python': r'def\s+(\w+)\s*\(',
        'go': r'func\s+(\w+)\s*\(',
        'java': r'(?:public|private|protected)?\s+(?:static\s+)?[\w<>]+\s+(\w+)\s*\(',
        'csharp': r'(?:public|private|protected)?\s+(?:static\s+)?[\w<>]+\s+(\w+)\s*\(',
        'php': r'function\s+(\w+)\s*\(',
        'ruby': r'def\s+(\w+)'
    }
    
    pattern = patterns.get(language, r'function\s+(\w+)\s*\(')
    matches = re.finditer(pattern, content, re.MULTILINE)
    
    for match in matches:
        line_num = content[:match.start()].count('\n') + 1
        func_name = match.group(1)
        
        # Find function end (simplified - find next function or end of file)
        next_match = None
        for m in re.finditer(pattern, content[match.end():], re.MULTILINE):
            next_match = m
            break
        
        end_pos = match.start() + match.end() + next_match.start() if next_match else len(content)
        
        functions.append({
            'name': func_name,
            'line': line_num,
            'start': match.start(),
            'end': end_pos
        })
    
    return functions


def scan_phase_1_apis_only(repo_root: str) -> Dict:
    """Phase 1: Scan for APIs only (internal + external)"""
    print(f"[Phase 1] Scanning for APIs only: {repo_root}")
    
    code_files = find_code_files(repo_root)
    print(f"[Phase 1] Found {len(code_files)} code files")
    
    internal_apis = []
    external_apis = []
    
    for file_info in code_files:
        try:
            with open(file_info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language = file_info['language']
            
            # Detect internal APIs
            file_internal = detect_internal_apis(content, language, file_info['path'])
            internal_apis.extend(file_internal)
            
            # Detect external APIs
            file_external = detect_external_apis(content, language, file_info['path'])
            external_apis.extend(file_external)
        
        except Exception as e:
            continue
    
    # Aggregate and deduplicate
    internal_aggregated = aggregate_internal_apis(internal_apis)
    external_aggregated = aggregate_external_apis(external_apis)
    
    print(f"[Phase 1] Found {len(internal_aggregated)} internal APIs")
    print(f"[Phase 1] Found {len(external_aggregated)} external APIs")
    
    return {
        'phase': 1,
        'repo_root': repo_root,
        'internal_apis': internal_aggregated,
        'external_apis': external_aggregated,
        'summary': {
            'internal_api_count': len(internal_aggregated),
            'external_api_count': len(external_aggregated),
            'total_api_count': len(internal_aggregated) + len(external_aggregated)
        }
    }


def scan_phase_2_services(repo_root: str, selected_apis: List[str]) -> Dict:
    """Phase 2: Find services using selected APIs"""
    print(f"[Phase 2] Finding services using {len(selected_apis)} selected API(s)")
    
    code_files = find_code_files(repo_root)
    
    services_by_api = {}
    
    for api_spec in selected_apis:
        # Parse api_spec: format is "type:api_id" (e.g., "internal:GET /api/users" or "external:OpenAI API")
        if ':' in api_spec:
            api_type, api_id = api_spec.split(':', 1)
        else:
            # Default to external if no type specified
            api_type = 'external'
            api_id = api_spec
        
        print(f"[Phase 2] Scanning for services using {api_type} API: {api_id}")
        services = find_services_using_api(code_files, api_id, api_type, repo_root)
        
        services_by_api[api_spec] = {
            'api_type': api_type,
            'api_id': api_id,
            'services': services,
            'service_count': len(services)
        }
        
        print(f"[Phase 2] Found {len(services)} services using {api_id}")
    
    total_services = sum(len(api_data['services']) for api_data in services_by_api.values())
    
    return {
        'phase': 2,
        'repo_root': repo_root,
        'selected_apis': selected_apis,
        'services_by_api': services_by_api,
        'summary': {
            'apis_analyzed': len(selected_apis),
            'total_services_found': total_services
        }
    }


def scan_codebase(repo_root: str = None, phase: int = 1, selected_apis: List[str] = None) -> Dict:
    """
    Two-phase codebase scanner
    Phase 1: Scan for APIs only (fast)
    Phase 2: Scan for services using selected APIs (targeted)
    """
    if repo_root is None:
        repo_root = os.getcwd()
    
    repo_root = os.path.abspath(repo_root)
    
    if phase == 1:
        return scan_phase_1_apis_only(repo_root)
    elif phase == 2:
        if not selected_apis:
            print("[ERROR] Phase 2 requires --selected-apis parameter")
            return {'error': 'Phase 2 requires selected APIs'}
        return scan_phase_2_services(repo_root, selected_apis)
    else:
        print(f"[ERROR] Invalid phase: {phase}. Must be 1 or 2.")
        return {'error': f'Invalid phase: {phase}'}


def scan_codebase_legacy(repo_root: str = None) -> Dict:
    """Legacy scan - full scan (kept for backward compatibility)"""
    if repo_root is None:
        repo_root = os.getcwd()
    
    repo_root = os.path.abspath(repo_root)
    
    print(f"Scanning codebase: {repo_root}")
    
    # Find all code files
    code_files = find_code_files(repo_root)
    print(f"Found {len(code_files)} code files")
    
    # Scan for Temporal implementations
    temporal_found = False
    workflows_detected = []
    activities_detected = []
    workers_detected = []
    temporal_files = []
    
    # Scan for patterns
    all_patterns = {
        'long_running': [],
        'multi_step': [],
        'external_api': [],
        'background_jobs': [],
        'scheduled_tasks': []
    }
    
    # NEW: Scan for external APIs
    all_apis = []
    
    for file_info in code_files:
        try:
            with open(file_info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language = file_info['language']
            
            # Check for Temporal imports
            if detect_temporal_imports(content, language):
                temporal_found = True
                temporal_files.append(file_info['path'])
                
                # Detect workflows
                workflows = detect_workflow_definitions(content, language)
                workflows_detected.extend([{
                    'name': wf,
                    'file': file_info['path'],
                    'language': language
                } for wf in workflows])
                
                # Detect activities
                activities = detect_activity_definitions(content, language)
                activities_detected.extend([{
                    'name': act,
                    'file': file_info['path'],
                    'language': language
                } for act in activities])
                
                # Detect workers
                if detect_worker_setups(content, language):
                    workers_detected.append({
                        'file': file_info['path'],
                        'language': language
                    })
            
            # Detect patterns
            file_patterns = detect_patterns(content, language)
            for pattern_type, matches in file_patterns.items():
                for match in matches:
                    match['file'] = file_info['path']
                    match['language'] = language
                    all_patterns[pattern_type].append(match)
            
            # NEW: Detect external APIs
            file_apis = detect_external_apis(content, language, file_info['path'])
            all_apis.extend(file_apis)
        
        except Exception as e:
            # Skip files that can't be read
            continue
    
    # NEW: Aggregate API statistics
    api_stats = {}
    for api_entry in all_apis:
        api_name = api_entry['api_name']
        if api_name not in api_stats:
            api_stats[api_name] = {
                'name': api_name,
                'call_count': 0,
                'file_count': 0,
                'files': set(),
                'usages': []
            }
        api_stats[api_name]['call_count'] += 1
        api_stats[api_name]['files'].add(api_entry['file'])
        api_stats[api_name]['usages'].append({
            'file': api_entry['file'],
            'line': api_entry['line'],
            'method': api_entry['detection_method'],
            'context': api_entry.get('context', '')
        })
    
    # Convert sets to lists for JSON serialization
    for api_name in api_stats:
        api_stats[api_name]['files'] = list(api_stats[api_name]['files'])
        api_stats[api_name]['file_count'] = len(api_stats[api_name]['files'])
    
    # Sort APIs by call count (descending)
    sorted_apis = sorted(api_stats.values(), key=lambda x: x['call_count'], reverse=True)
    
    # Generate report
    report = {
        'repo_root': repo_root,
        'scan_timestamp': str(Path(__file__).stat().st_mtime),
        'files_scanned': len(code_files),
        'temporal': {
            'found': temporal_found,
            'files': temporal_files,
            'workflows': workflows_detected,
            'activities': activities_detected,
            'workers': workers_detected
        },
        'patterns': all_patterns,
        'apis': sorted_apis,  # NEW: Sorted by call count
        'summary': {
            'temporal_found': temporal_found,
            'workflows_count': len(workflows_detected),
            'activities_count': len(activities_detected),
            'workers_count': len(workers_detected),
            'patterns_found': sum(len(v) for v in all_patterns.values()),
            'apis_found': len(api_stats)  # NEW
        }
    }
    
    return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Two-phase codebase scanner for Temporal integration')
    parser.add_argument('repo_root', nargs='?', default=os.getcwd(),
                        help='Repository root path (default: current directory)')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=1,
                        help='Scan phase: 1=APIs only (fast), 2=services using selected APIs (targeted)')
    parser.add_argument('--selected-apis', type=str, default='',
                        help='Comma-separated API identifiers for phase 2 (format: type:api_id)')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy full scan (backward compatibility)')
    
    args = parser.parse_args()
    
    try:
        if args.legacy:
            # Legacy mode - full scan
            report = scan_codebase_legacy(args.repo_root)
        else:
            # Two-phase mode
            selected_apis = None
            if args.selected_apis:
                selected_apis = [api.strip() for api in args.selected_apis.split(',') if api.strip()]
            
            report = scan_codebase(args.repo_root, args.phase, selected_apis)
        
        # Save to JSON file
        output_file = 'temporal_scan_results.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Scan results saved to: {output_file}")
        
        # Also output to stdout for debugging
        print("\n" + "="*60)
        print(f"SCAN REPORT - Phase {report.get('phase', 'Legacy')}")
        print("="*60)
        print(json.dumps(report, indent=2))
        
        return 0
    except Exception as e:
        print(f"Error scanning codebase: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

