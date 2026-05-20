#!/usr/bin/env python3
"""
DEEP CODEBASE SCANNER - Finds ALL API usage across entire codebase
Scans: Controllers → Services → Repositories → Async processes → Nested calls
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set

# Exclude patterns
EXCLUDE = ['node_modules', '.git', 'venv', '__pycache__', 'build', 'dist', 'target']

def should_scan(file_path: str) -> bool:
    """Check if file should be scanned"""
    return not any(ex in file_path for ex in EXCLUDE) and file_path.endswith(('.java', '.py', '.ts', '.js', '.go'))

def extract_imports(content: str, language: str) -> List[str]:
    """Extract all imports from file"""
    imports = []
    if language == 'java':
        imports = re.findall(r'import\s+([\w.]+)', content)
    elif language in ['python', 'py']:
        imports = re.findall(r'from\s+([\w.]+)\s+import|import\s+([\w.]+)', content)
    elif language in ['typescript', 'javascript', 'ts', 'js']:
        imports = re.findall(r'from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\)', content)
    return [i[0] if isinstance(i, tuple) else i for i in imports if i]

def find_method_calls(content: str, language: str) -> List[Dict]:
    """Find all method/function calls in file"""
    calls = []
    
    if language == 'java':
        # Find service/repository injections
        injections = re.findall(r'@Autowired\s+(?:private|public)?\s+(\w+)\s+(\w+)', content)
        for class_name, var_name in injections:
            calls.append({'type': 'injection', 'class': class_name, 'variable': var_name})
        
        # Find method calls
        method_calls = re.findall(r'(\w+)\.(\w+)\s*\([^)]*\)', content)
        for obj, method in method_calls:
            calls.append({'type': 'method_call', 'object': obj, 'method': method})
        
        # Find async annotations
        if '@Async' in content:
            async_methods = re.findall(r'@Async[^\n]*\n[^\n]*(?:public|private)\s+\w+\s+(\w+)\s*\(', content)
            for method in async_methods:
                calls.append({'type': 'async_method', 'method': method})
    
    elif language in ['python', 'py']:
        # Find function calls
        func_calls = re.findall(r'(\w+)\.(\w+)\s*\([^)]*\)|(\w+)\s*\([^)]*\)', content)
        for call in func_calls:
            if call[0]:  # obj.method()
                calls.append({'type': 'method_call', 'object': call[0], 'method': call[1]})
            elif call[2]:  # function()
                calls.append({'type': 'function_call', 'function': call[2]})
    
    return calls

def find_api_endpoints(content: str, language: str) -> List[Dict]:
    """Find API endpoint definitions"""
    endpoints = []
    
    if language == 'java':
        # Spring annotations
        patterns = [
            (r'@(?:Get|Post|Put|Delete|Patch)Mapping\s*\(["\']([^"\']+)["\']\)', 'REST'),
            (r'@RequestMapping\s*\([^)]*value\s*=\s*["\']([^"\']+)["\']\)', 'REST'),
            (r'@(?:Query|Mutation)\s*\(["\']([^"\']+)["\']\)', 'GraphQL')
        ]
        
        for pattern, api_type in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Find the method this annotation is on
                method_match = re.search(rf'{pattern}[^{{]*(\w+)\s*\(', content)
                if method_match:
                    endpoints.append({
                        'type': api_type,
                        'path': match,
                        'method_name': method_match.group(2) if len(method_match.groups()) > 1 else 'unknown'
                    })
    
    return endpoints

def find_external_api_calls(content: str) -> List[Dict]:
    """Find calls to external APIs"""
    external = []
    
    # HTTP clients
    if 'RestTemplate' in content or 'WebClient' in content or 'HttpClient' in content:
        urls = re.findall(r'["\']https?://([^"\']+)["\']', content)
        for url in urls:
            external.append({'type': 'HTTP', 'url': url})
    
    # Known external APIs
    external_apis = {
        'openai': ['OpenAI', 'ChatCompletion', 'gpt-'],
        'stripe': ['Stripe', 'PaymentIntent', 'Customer'],
        'aws': ['AmazonS3', 'DynamoDB', 'SQS', 'SNS'],
        'twilio': ['Twilio', 'Messages.create'],
        'sendgrid': ['SendGrid', 'Mail'],
    }
    
    for api_name, patterns in external_apis.items():
        for pattern in patterns:
            if pattern in content:
                external.append({'type': 'external_api', 'name': api_name, 'pattern': pattern})
    
    return external

def deep_scan_file(file_path: str) -> Dict:
    """Deeply scan a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        ext = Path(file_path).suffix
        language = {'.java': 'java', '.py': 'python', '.ts': 'typescript', '.js': 'javascript'}.get(ext, 'unknown')
        
        return {
            'file': file_path,
            'language': language,
            'imports': extract_imports(content, language),
            'method_calls': find_method_calls(content, language),
            'api_endpoints': find_api_endpoints(content, language),
            'external_apis': find_external_api_calls(content),
            'has_async': '@Async' in content or 'async def' in content or 'async function' in content,
            'lines': len(content.splitlines())
        }
    except Exception as e:
        return {'file': file_path, 'error': str(e)}

def build_dependency_graph(scan_results: List[Dict]) -> Dict:
    """Build dependency graph showing what calls what"""
    graph = {
        'files': {},
        'call_chains': [],
        'api_to_services': {},
        'service_dependencies': {}
    }
    
    # Index all files
    for result in scan_results:
        if 'error' in result:
            continue
        
        file_name = Path(result['file']).stem
        graph['files'][file_name] = result
        
        # Map API endpoints to files
        for endpoint in result.get('api_endpoints', []):
            api_key = f"{endpoint['path']}"
            if api_key not in graph['api_to_services']:
                graph['api_to_services'][api_key] = []
            graph['api_to_services'][api_key].append({
                'file': result['file'],
                'method': endpoint['method_name']
            })
        
        # Map method calls to dependencies
        for call in result.get('method_calls', []):
            if call['type'] == 'injection':
                service = call['class']
                if file_name not in graph['service_dependencies']:
                    graph['service_dependencies'][file_name] = []
                graph['service_dependencies'][file_name].append(service)
    
    return graph

def trace_api_call_chain(api_endpoint: str, graph: Dict, max_depth: int = 5) -> List[str]:
    """Trace complete call chain from API endpoint through all services"""
    chain = []
    visited = set()
    
    def trace(node, depth):
        if depth > max_depth or node in visited:
            return
        visited.add(node)
        chain.append(node)
        
        # Find dependencies
        if node in graph['service_dependencies']:
            for dep in graph['service_dependencies'][node]:
                trace(dep, depth + 1)
    
    # Start from API endpoint
    if api_endpoint in graph['api_to_services']:
        for service_info in graph['api_to_services'][api_endpoint]:
            file_name = Path(service_info['file']).stem
            trace(file_name, 0)
    
    return chain

def deep_scan_codebase(repo_root: str) -> Dict:
    """Perform deep scan of entire codebase"""
    print(f"[DEEP SCAN] Scanning: {repo_root}")
    
    # Find all code files
    all_files = []
    for root, dirs, files in os.walk(repo_root):
        # Remove excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE]
        
        for file in files:
            file_path = os.path.join(root, file)
            if should_scan(file_path):
                all_files.append(file_path)
    
    print(f"[DEEP SCAN] Found {len(all_files)} code files")
    
    # Scan each file
    scan_results = []
    for i, file_path in enumerate(all_files):
        if i % 50 == 0:
            print(f"[DEEP SCAN] Processing file {i+1}/{len(all_files)}...")
        result = deep_scan_file(file_path)
        scan_results.append(result)
    
    print(f"[DEEP SCAN] Building dependency graph...")
    graph = build_dependency_graph(scan_results)
    
    # Analyze results
    total_apis = len(graph['api_to_services'])
    total_services = len(graph['service_dependencies'])
    total_external = sum(len(r.get('external_apis', [])) for r in scan_results if 'error' not in r)
    async_files = sum(1 for r in scan_results if r.get('has_async', False))
    
    print(f"\n[DEEP SCAN] Results:")
    print(f"  - API Endpoints: {total_apis}")
    print(f"  - Services: {total_services}")
    print(f"  - External APIs: {total_external}")
    print(f"  - Async processes: {async_files}")
    
    # Build complete picture
    complete_analysis = {
        'repo_root': repo_root,
        'total_files_scanned': len(all_files),
        'api_endpoints': list(graph['api_to_services'].keys()),
        'services': list(graph['service_dependencies'].keys()),
        'dependency_graph': graph,
        'scan_results': scan_results,
        'summary': {
            'total_apis': total_apis,
            'total_services': total_services,
            'total_external_apis': total_external,
            'async_files': async_files
        }
    }
    
    # Trace call chains for each API
    print(f"\n[DEEP SCAN] Tracing call chains...")
    for api in list(graph['api_to_services'].keys())[:10]:  # Sample first 10
        chain = trace_api_call_chain(api, graph)
        if chain:
            print(f"  {api} → {' → '.join(chain[:5])}")
    
    return complete_analysis

if __name__ == '__main__':
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    result = deep_scan_codebase(repo)
    
    # Save to file
    output_file = 'deep-scan-results.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[DEEP SCAN] Results saved to: {output_file}")
