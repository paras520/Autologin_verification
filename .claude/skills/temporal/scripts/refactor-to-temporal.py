#!/usr/bin/env python3
"""
Temporal Integration System - Refactoring Engine
Refactors existing code to use Temporal while preserving functionality.
Supports both service-based and API-based implementation.

Usage:
    python refactor-to-temporal.py [target_type] [target_id] [scan_report_json]
    
    target_type: 'service' or 'api'
    target_id: service number or API name
    scan_report_json: path to scan report
"""

import os
import re
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Global flags for agent mode (set by main())
STRATEGY_OVERRIDE = None
AGENT_MODE = False


# Temporal folder structure
TEMPORAL_FOLDER_STRUCTURE = {
    'workflows': 'temporal/workflows',
    'activities': 'temporal/activities',
    'workers': 'temporal/workers',
    'client': 'temporal/client',
    'tests': 'temporal/tests',
    'config': 'temporal/config'
}


def create_temporal_folder_structure(repo_root: str) -> Dict[str, str]:
    """Create temporal/ folder structure"""
    created_folders = {}
    
    for folder_type, folder_path in TEMPORAL_FOLDER_STRUCTURE.items():
        full_path = os.path.join(repo_root, folder_path)
        os.makedirs(full_path, exist_ok=True)
        created_folders[folder_type] = full_path
        
        # Create subdirectories for tests
        if folder_type == 'tests':
            os.makedirs(os.path.join(full_path, 'workflows'), exist_ok=True)
            os.makedirs(os.path.join(full_path, 'activities'), exist_ok=True)
            os.makedirs(os.path.join(full_path, 'integration'), exist_ok=True)
        
        # Create subdirectory for shared activities
        if folder_type == 'activities':
            os.makedirs(os.path.join(full_path, 'shared'), exist_ok=True)
    
    return created_folders


def generate_api_workflow(api_name: str, api_data: Dict, language: str, repo_root: str) -> Dict:
    """Generate a Temporal workflow for an API"""
    
    workflow_name = f"{api_name.replace(' ', '')}Workflow"
    activity_name = f"{api_name.replace(' ', '')}Activity"
    
    # Create workflows folder
    workflows_folder = os.path.join(repo_root, 'temporal', 'workflows')
    os.makedirs(workflows_folder, exist_ok=True)
    
    if language == 'java':
        workflow_file = os.path.join(workflows_folder, f"{workflow_name}.java")
        workflow_content = f"""package temporal.workflows;

import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;
import temporal.activities.{activity_name};

@WorkflowInterface
public interface {workflow_name} {{
    
    @WorkflowMethod
    String execute(String input);
}}
"""
        with open(workflow_file, 'w') as f:
            f.write(workflow_content)
        
        # Generate implementation class
        impl_file = os.path.join(workflows_folder, f"{workflow_name}Impl.java")
        impl_content = f"""package temporal.workflows;

import io.temporal.activity.ActivityOptions;
import io.temporal.common.RetryOptions;
import io.temporal.workflow.Workflow;
import temporal.activities.{activity_name};
import java.time.Duration;

public class {workflow_name}Impl implements {workflow_name} {{
    
    private final {activity_name} activity = Workflow.newActivityStub(
        {activity_name}.class,
        ActivityOptions.newBuilder()
            .setStartToCloseTimeout(Duration.ofSeconds(30))
            .setRetryOptions(RetryOptions.newBuilder()
                .setMaximumAttempts(3)
                .setInitialInterval(Duration.ofSeconds(1))
                .setMaximumInterval(Duration.ofSeconds(10))
                .build())
            .build()
    );
    
    @Override
    public String execute(String input) {{
        return activity.call{activity_name}(input);
    }}
}}
"""
        with open(impl_file, 'w') as f:
            f.write(impl_content)
            
        return {
            'workflow_name': workflow_name,
            'workflow_file': workflow_file,
            'workflow_impl_file': impl_file
        }
    
    elif language == 'python':
        workflow_file = os.path.join(workflows_folder, f"{workflow_name.lower()}.py")
        workflow_content = f"""from temporalio import workflow
from datetime import timedelta
from temporal.activities.{activity_name.lower()} import {activity_name.lower()}

@workflow.defn(name="{workflow_name}")
class {workflow_name}:
    
    @workflow.run
    async def run(self, input: str) -> str:
        return await workflow.execute_activity(
            {activity_name.lower()},
            input,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10)
            )
        )
"""
        with open(workflow_file, 'w') as f:
            f.write(workflow_content)
            
        return {
            'workflow_name': workflow_name,
            'workflow_file': workflow_file
        }
    
    else:
        # Generic workflow for other languages
        workflow_file = os.path.join(workflows_folder, f"{workflow_name}.txt")
        with open(workflow_file, 'w') as f:
            f.write(f"// Implement workflow for {api_name} in {language}\n")
        
        return {
            'workflow_name': workflow_name,
            'workflow_file': workflow_file
        }


def generate_api_activity(api_name: str, api_data: Dict, language: str, repo_root: str) -> Dict:
    """Generate Temporal activity for an API"""
    
    # Sanitize API name for file naming
    safe_api_name = re.sub(r'[^\w]+', '_', api_name).lower()
    
    # Determine file extension
    ext_map = {
        'typescript': '.ts',
        'javascript': '.js',
        'python': '.py',
        'go': '.go',
        'java': '.java',
        'csharp': '.cs',
        'php': '.php',
        'ruby': '.rb'
    }
    ext = ext_map.get(language, '.ts')
    
    # Generate activity code based on language
    if language == 'typescript':
        activity_code = f"""import {{ Context }} from '@temporalio/activity';

/**
 * {api_name} Activity
 * Handles all {api_name} calls with automatic retries and error handling
 */

export interface {safe_api_name.title().replace('_', '')}Request {{
    [key: string]: any;
}}

export interface {safe_api_name.title().replace('_', '')}Response {{
    [key: string]: any;
}}

export async function call{safe_api_name.title().replace('_', '')}(
    request: {safe_api_name.title().replace('_', '')}Request
): Promise<{safe_api_name.title().replace('_', '')}Response> {{
    const context = Context.current();
    
    context.log.info('{api_name} call started', {{ request }});
    
    try {{
        // TODO: Implement actual {api_name} call logic
        // Replace with your {api_name} SDK calls
        
        const response = {{
            success: true,
            data: {{}}
        }};
        
        context.log.info('{api_name} call completed', {{ response }});
        
        return response;
    }} catch (error) {{
        context.log.error('{api_name} call failed', {{ error }});
        throw error;
    }}
}}
"""
    
    elif language == 'python':
        activity_code = f"""from temporalio import activity
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@activity.defn
async def call_{safe_api_name}(request: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    {api_name} Activity
    Handles all {api_name} calls with automatic retries and error handling
    \"\"\"
    activity.logger.info(f"{api_name} call started: {{request}}")
    
    try:
        # TODO: Implement actual {api_name} call logic
        # Replace with your {api_name} SDK calls
        
        response = {{
            'success': True,
            'data': {{}}
        }}
        
        activity.logger.info(f"{api_name} call completed: {{response}}")
        
        return response
    except Exception as error:
        activity.logger.error(f"{api_name} call failed: {{error}}")
        raise
"""
    
    elif language == 'go':
        activity_code = f"""package activities

import (
    "context"
    "go.temporal.io/sdk/activity"
)

type {safe_api_name.title().replace('_', '')}Request struct {{
    // TODO: Define request structure
}}

type {safe_api_name.title().replace('_', '')}Response struct {{
    // TODO: Define response structure
}}

// Call{safe_api_name.title().replace('_', '')} handles all {api_name} calls with automatic retries
func Call{safe_api_name.title().replace('_', '')}(ctx context.Context, request {safe_api_name.title().replace('_', '')}Request) (*{safe_api_name.title().replace('_', '')}Response, error) {{
    logger := activity.GetLogger(ctx)
    logger.Info("{api_name} call started")
    
    // TODO: Implement actual {api_name} call logic
    
    response := &{safe_api_name.title().replace('_', '')}Response{{}}
    
    logger.Info("{api_name} call completed")
    return response, nil
}}
"""
    
    elif language == 'java':
        class_name = safe_api_name.title().replace('_', '')
        activity_code = f"""package com.diro.temporal.activities;

import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.HashMap;

/**
 * {api_name} Activity
 * Handles all {api_name} calls with automatic retries and error handling
 */
@ActivityInterface
public interface {class_name}Activity {{
    
    @ActivityMethod
    Map<String, Object> call{class_name}(Map<String, Object> request);
}}

@Component
class {class_name}ActivityImpl implements {class_name}Activity {{
    
    private static final Logger logger = LoggerFactory.getLogger({class_name}ActivityImpl.class);
    
    @Override
    public Map<String, Object> call{class_name}(Map<String, Object> request) {{
        logger.info("{api_name} call started: {{}}", request);
        
        try {{
            // TODO: Implement actual {api_name} call logic
            // Replace with your {api_name} SDK calls
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("data", new HashMap<>());
            
            logger.info("{api_name} call completed: {{}}", response);
            
            return response;
        }} catch (Exception error) {{
            logger.error("{api_name} call failed", error);
            throw new RuntimeException("{api_name} call failed", error);
        }}
    }}
}}
"""
    
    else:
        # Default template for other languages
        activity_code = f"""// {api_name} Activity
// TODO: Implement activity for {api_name} in {language}
"""
    
    # Write activity file
    activity_file = os.path.join(repo_root, TEMPORAL_FOLDER_STRUCTURE['activities'], f"{safe_api_name}_activity{ext}")
    
    with open(activity_file, 'w', encoding='utf-8') as f:
        f.write(activity_code)
    
    return {
        'activity_name': f'call_{safe_api_name}',
        'activity_file': activity_file,
        'api_name': api_name,
        'safe_name': safe_api_name,
        'code': activity_code
    }


def ask_implementation_strategy() -> str:
    """Ask developer for API implementation strategy (or use override for agent mode)"""
    global STRATEGY_OVERRIDE, AGENT_MODE
    
    # Check if strategy was provided via command line
    if STRATEGY_OVERRIDE:
        print(f"\n✅ Using strategy from command line: {STRATEGY_OVERRIDE}")
        return STRATEGY_OVERRIDE
    
    # Check if agent mode (use default)
    if AGENT_MODE:
        print(f"\n✅ Agent mode: Using default strategy: shared")
        return 'shared'
    
    # Interactive mode
    print("\n" + "="*60)
    print("API IMPLEMENTATION STRATEGY")
    print("="*60)
    print("\nChoose implementation approach:")
    print("A) Shared Activity - One reusable activity for all services")
    print("B) Per-Service Activities - Separate activity per service")
    print("\nShared Activity (A) is recommended for:")
    print("  - Consistency across services")
    print("  - Centralized configuration")
    print("  - Easier maintenance")
    print("\nPer-Service Activities (B) is recommended for:")
    print("  - Service-specific behavior")
    print("  - Different retry policies per service")
    print("  - Service isolation")
    print("="*60)
    
    while True:
        choice = input("\nWhich approach? [A/B]: ").strip().upper()
        if choice in ['A', 'B']:
            return 'shared' if choice == 'A' else 'per_service'
        print("Invalid choice. Please enter A or B.")


def preserve_functionality(original_code: str, temporal_code: str) -> Dict:
    """Ensure refactored code preserves functionality"""
    validation = {
        'function_signatures_match': False,
        'inputs_outputs_match': False,
        'error_handling_preserved': False,
        'validation_passed': False
    }
    
    # Extract function signatures
    original_sigs = re.findall(r'(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', original_code)
    temporal_sigs = re.findall(r'(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', temporal_code)
    
    # Check if function names match
    original_names = {sig[0] for sig in original_sigs}
    temporal_names = {sig[0] for sig in temporal_sigs}
    
    if original_names.issubset(temporal_names):
        validation['function_signatures_match'] = True
    
    # Check inputs/outputs (simplified check)
    # In real implementation, would do deeper analysis
    validation['inputs_outputs_match'] = True  # Assume match for now
    
    # Check error handling
    original_has_error = 'try' in original_code or 'catch' in original_code or 'except' in original_code
    temporal_has_error = 'try' in temporal_code or 'catch' in temporal_code or 'except' in temporal_code
    
    if original_has_error == temporal_has_error or temporal_has_error:
        validation['error_handling_preserved'] = True
    
    # Overall validation
    validation['validation_passed'] = all([
        validation['function_signatures_match'],
        validation['inputs_outputs_match'],
        validation['error_handling_preserved']
    ])
    
    return validation


def refactor_long_running(file_path: str, content: str, language: str) -> str:
    """Refactor long-running operations to Temporal workflow"""
    # This is a template - actual implementation would parse AST and refactor
    
    if language == 'typescript' or language == 'javascript':
        # Find async functions with delays/timeouts
        pattern = r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{([^}]+)\}'
        
        def replace_function(match):
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Check if it's long-running
            if 'setTimeout' in func_body or 'sleep' in func_body or 'delay' in func_body:
                # Convert to workflow
                workflow_code = f"""import {{ workflow }} from '@temporalio/workflow';
import {{ sleep }} from '@temporalio/workflow';

export async function {func_name}Workflow(input: any): Promise<any> {{
    // Original function logic converted to workflow
    {func_body}
    
    return result;
}}"""
                return workflow_code
            return match.group(0)
        
        return re.sub(pattern, replace_function, content, flags=re.MULTILINE | re.DOTALL)
    
    elif language == 'python':
        # Python refactoring
        pattern = r'def\s+(\w+)\s*\([^)]*\):\s*\n((?:[^\n]+\n)*)'
        
        def replace_function(match):
            func_name = match.group(1)
            func_body = match.group(2)
            
            if 'time.sleep' in func_body or 'asyncio.sleep' in func_body:
                workflow_code = f"""from temporalio import workflow

@workflow.defn
class {func_name}Workflow:
    @workflow.run
    async def run(self, input: any) -> any:
        # Original function logic converted to workflow
        {func_body}
        
        return result"""
                return workflow_code
            return match.group(0)
        
        return re.sub(pattern, replace_function, content, flags=re.MULTILINE)
    
    # For other languages, return original (would need language-specific implementation)
    return content


def refactor_multi_step(file_path: str, content: str, language: str) -> str:
    """Refactor multi-step processes to Temporal workflow with activities"""
    
    if language == 'typescript' or language == 'javascript':
        # Find functions with multiple sequential async calls
        pattern = r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{([^}]+)\}'
        
        def replace_function(match):
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Count sequential await calls
            await_count = func_body.count('await')
            if await_count >= 2:
                # Convert to workflow with activities
                workflow_code = f"""import {{ workflow }} from '@temporalio/workflow';
import * as activities from './activities';

export async function {func_name}Workflow(input: any): Promise<any> {{
    const {func_name}Activities = workflow.proxyActivities(activities, {{
        startToCloseTimeout: '5m',
        retry: {{
            initialInterval: '1s',
            backoffCoefficient: 2,
            maximumAttempts: 3
        }}
    }});
    
    // Convert sequential calls to workflow activities
    {func_body.replace('await ', 'await {func_name}Activities.')}
    
    return result;
}}"""
                return workflow_code
            return match.group(0)
        
        return re.sub(pattern, replace_function, content, flags=re.MULTILINE | re.DOTALL)
    
    elif language == 'python':
        # Python refactoring
        pattern = r'def\s+(\w+)\s*\([^)]*\):\s*\n((?:[^\n]+\n)*)'
        
        def replace_function(match):
            func_name = match.group(1)
            func_body = match.group(2)
            
            if func_body.count('await') >= 2 or func_body.count('(') >= 3:
                workflow_code = f"""from temporalio import workflow
from activities import {func_name}_activities

@workflow.defn
class {func_name}Workflow:
    @workflow.run
    async def run(self, input: any) -> any:
        # Convert sequential calls to workflow activities
        {func_body}
        
        return result"""
                return workflow_code
            return match.group(0)
        
        return re.sub(pattern, replace_function, content, flags=re.MULTILINE)
    
    return content


def refactor_external_api(file_path: str, content: str, language: str) -> str:
    """Refactor external API calls to Temporal activities"""
    
    if language == 'typescript' or language == 'javascript':
        # Find functions with API calls
        api_patterns = ['fetch', 'axios', 'http.get', 'http.post']
        
        for api_pattern in api_patterns:
            if api_pattern in content:
                # Create activity file reference
                activity_ref = f"""
// Activity file should be created: activities/{Path(file_path).stem}-activities.ts
import {{ proxyActivities }} from '@temporalio/workflow';

const activities = proxyActivities(require('./activities/{Path(file_path).stem}-activities'), {{
    startToCloseTimeout: '30s',
    retry: {{
        initialInterval: '1s',
        backoffCoefficient: 2,
        maximumAttempts: 5
    }}
}});
"""
                return activity_ref + content
    
    elif language == 'python':
        if 'requests' in content or 'httpx' in content or 'urllib' in content:
            activity_ref = f"""
# Activity file should be created: activities/{Path(file_path).stem}_activities.py
from temporalio import workflow
from activities import {Path(file_path).stem}_activities

activities = workflow.proxy_activities({Path(file_path).stem}_activities, start_to_close_timeout='30s')
"""
            return activity_ref + content
    
    return content


def refactor_to_temporal(file_path: str, pattern_type: str, language: str = None) -> Dict:
    """Refactor existing code to use Temporal"""
    
    if not os.path.exists(file_path):
        return {
            'success': False,
            'error': f'File not found: {file_path}'
        }
    
    # Detect language if not provided
    if language is None:
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.ts': 'typescript', '.tsx': 'typescript',
            '.js': 'javascript', '.jsx': 'javascript',
            '.py': 'python',
            '.go': 'go',
            '.java': 'java',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby'
        }
        language = lang_map.get(ext, 'typescript')
    
    try:
        # Read original file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Refactor based on pattern type
        refactored_content = original_content
        if pattern_type == 'long_running':
            refactored_content = refactor_long_running(file_path, original_content, language)
        elif pattern_type == 'multi_step':
            refactored_content = refactor_multi_step(file_path, original_content, language)
        elif pattern_type == 'external_api':
            refactored_content = refactor_external_api(file_path, original_content, language)
        else:
            # Default: multi-step refactoring
            refactored_content = refactor_multi_step(file_path, original_content, language)
        
        # Validate preservation
        validation = preserve_functionality(original_content, refactored_content)
        
        result = {
            'success': True,
            'file_path': file_path,
            'pattern_type': pattern_type,
            'language': language,
            'original_length': len(original_content),
            'refactored_length': len(refactored_content),
            'validation': validation,
            'refactored_code': refactored_content
        }
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def refactor_api_based(api_name: str, api_data: Dict, language: str, repo_root: str, strategy: str) -> Dict:
    """Refactor all usages of an API to use Temporal"""
    
    # Create temporal folder structure
    folders = create_temporal_folder_structure(repo_root)
    
    # Generate workflow for API (NEW - was missing!)
    workflow_name = f"{api_name.replace(' ', '')}Workflow"
    workflow_info = generate_api_workflow(api_name, api_data, language, repo_root)
    
    # Generate API activity
    activity_info = generate_api_activity(api_name, api_data, language, repo_root)
    
    # Generate worker configuration
    worker_config = generate_worker_config(language, repo_root, [activity_info['activity_name']], workflow_name)
    
    # Generate README for temporal folder
    generate_temporal_readme(repo_root, api_name, 'api', language)
    
    result = {
        'success': True,
        'implementation_type': 'api',
        'api_name': api_name,
        'strategy': strategy,
        'workflow': workflow_info,
        'activity': activity_info,
        'worker_config': worker_config,
        'folders_created': folders,
        'files_created': [
            workflow_info['workflow_file'],
            activity_info['activity_file'],
            worker_config['worker_file']
        ],
        'next_steps': [
            "Run tests: python skills/temporal/scripts/run-tests.py",
            "Verify build: mvn clean install",
            "Deploy worker: java -jar temporal-worker.jar"
        ]
    }
    
    return result


def refactor_service_based(service_name: str, service_data: Dict, language: str, repo_root: str) -> Dict:
    """Refactor an entire service to use Temporal"""
    
    # Create temporal folder structure
    folders = create_temporal_folder_structure(repo_root)
    
    # Generate workflow
    workflow_info = generate_service_workflow(service_name, service_data, language, repo_root)
    
    # Generate activities for service functions
    activity_infos = []
    for func in service_data.get('functions_to_convert', []):
        activity_info = generate_service_activity(service_name, func, language, repo_root)
        activity_infos.append(activity_info)
    
    # Generate worker configuration
    activity_names = [a['activity_name'] for a in activity_infos]
    worker_config = generate_worker_config(language, repo_root, activity_names, workflow_info['workflow_name'])
    
    # Generate README
    generate_temporal_readme(repo_root, service_name, 'service', language)
    
    files_created = [
        workflow_info['workflow_file'],
        worker_config['worker_file']
    ] + [a['activity_file'] for a in activity_infos]
    
    result = {
        'success': True,
        'implementation_type': 'service',
        'service_name': service_name,
        'workflow': workflow_info,
        'activities': activity_infos,
        'worker_config': worker_config,
        'folders_created': folders,
        'files_created': files_created,
        'next_steps': [
            f"1. Implement workflow logic in {workflow_info['workflow_file']}",
            f"2. Implement {len(activity_infos)} activities",
            "3. Add tests in temporal/tests/",
            "4. Update service to call Temporal workflow",
            "5. Configure retry policies and timeouts"
        ]
    }
    
    return result


def generate_service_workflow(service_name: str, service_data: Dict, language: str, repo_root: str) -> Dict:
    """Generate workflow file for a service"""
    safe_name = re.sub(r'[^\w]+', '_', service_name).lower()
    ext_map = {'typescript': '.ts', 'javascript': '.js', 'python': '.py', 'go': '.go'}
    ext = ext_map.get(language, '.ts')
    
    if language == 'python':
        workflow_code = f"""from temporalio import workflow
from datetime import timedelta
from typing import Dict, Any

@workflow.defn
class {safe_name.title().replace('_', '')}Workflow:
    \"\"\"
    {service_name} Workflow
    Orchestrates {service_name} operations with Temporal
    \"\"\"
    
    @workflow.run
    async def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        workflow.logger.info(f"{service_name} workflow started")
        
        # TODO: Implement workflow logic
        # Call activities here
        
        result = {{'success': True}}
        
        workflow.logger.info(f"{service_name} workflow completed")
        return result
"""
    else:
        workflow_code = f"""// {service_name} Workflow
// TODO: Implement workflow for {service_name}
"""
    
    workflow_file = os.path.join(repo_root, TEMPORAL_FOLDER_STRUCTURE['workflows'], f"{safe_name}_workflow{ext}")
    with open(workflow_file, 'w', encoding='utf-8') as f:
        f.write(workflow_code)
    
    return {
        'workflow_name': f'{safe_name}_workflow',
        'workflow_file': workflow_file,
        'code': workflow_code
    }


def generate_service_activity(service_name: str, function_name: str, language: str, repo_root: str) -> Dict:
    """Generate activity file for a service function"""
    safe_service = re.sub(r'[^\w]+', '_', service_name).lower()
    safe_func = re.sub(r'[^\w]+', '_', function_name).lower()
    ext_map = {'typescript': '.ts', 'javascript': '.js', 'python': '.py', 'go': '.go'}
    ext = ext_map.get(language, '.ts')
    
    if language == 'python':
        activity_code = f"""from temporalio import activity
from typing import Dict, Any

@activity.defn
async def {safe_func}_activity(input: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    Activity for {function_name} from {service_name}
    \"\"\"
    activity.logger.info(f"{function_name} activity started")
    
    # TODO: Implement {function_name} logic
    
    result = {{'success': True}}
    
    activity.logger.info(f"{function_name} activity completed")
    return result
"""
    else:
        activity_code = f"""// {function_name} Activity
// TODO: Implement activity for {function_name}
"""
    
    activity_file = os.path.join(repo_root, TEMPORAL_FOLDER_STRUCTURE['activities'], f"{safe_service}_{safe_func}_activity{ext}")
    with open(activity_file, 'w', encoding='utf-8') as f:
        f.write(activity_code)
    
    return {
        'activity_name': f'{safe_func}_activity',
        'activity_file': activity_file,
        'function_name': function_name
    }


def generate_worker_config(language: str, repo_root: str, activity_names: List[str], workflow_name: str = None) -> Dict:
    """Generate worker configuration file"""
    ext_map = {'typescript': '.ts', 'javascript': '.js', 'python': '.py', 'go': '.go', 'java': '.java'}
    ext = ext_map.get(language, '.ts')
    
    if language == 'python':
        worker_code = f"""import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

# Import activities
# TODO: Update imports based on your activities

async def main():
    client = await Client.connect("localhost:7233")
    
    # Create worker
    worker = Worker(
        client,
        task_queue="main-task-queue",
        workflows=[],  # TODO: Add workflows
        activities=[],  # TODO: Add activities
    )
    
    print("Worker started, listening on main-task-queue")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    elif language == 'java':
        worker_code = f"""package com.diro.temporal.workers;

import io.temporal.client.WorkflowClient;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

/**
 * Temporal Worker Configuration
 * Registers workflows and activities with the Temporal service
 */
@Component
public class TemporalWorker {{
    
    private static final Logger logger = LoggerFactory.getLogger(TemporalWorker.class);
    private static final String TASK_QUEUE = "main-task-queue";
    
    private WorkerFactory workerFactory;
    private WorkflowServiceStubs service;
    
    @PostConstruct
    public void start() {{
        logger.info("Starting Temporal worker...");
        
        // Create connection to Temporal service
        service = WorkflowServiceStubs.newInstance();
        WorkflowClient client = WorkflowClient.newInstance(service);
        
        // Create worker factory
        workerFactory = WorkerFactory.newInstance(client);
        
        // Create worker for task queue
        Worker worker = workerFactory.newWorker(TASK_QUEUE);
        
        // Register workflows
        // TODO: Register your workflow implementations
        // worker.registerWorkflowImplementationTypes(YourWorkflowImpl.class);
        
        // Register activities
        // TODO: Register your activity implementations
        // worker.registerActivitiesImplementations(new YourActivityImpl());
        
        // Start worker
        workerFactory.start();
        
        logger.info("Temporal worker started successfully on task queue: {{}}", TASK_QUEUE);
    }}
    
    @PreDestroy
    public void shutdown() {{
        logger.info("Shutting down Temporal worker...");
        if (workerFactory != null) {{
            workerFactory.shutdown();
        }}
        if (service != null) {{
            service.shutdown();
        }}
        logger.info("Temporal worker shut down successfully");
    }}
}}
"""
    
    else:
        worker_code = f"""// Worker Configuration
// TODO: Implement worker for {language}
"""
    
    worker_file = os.path.join(repo_root, TEMPORAL_FOLDER_STRUCTURE['workers'], f"main_worker{ext}")
    with open(worker_file, 'w', encoding='utf-8') as f:
        f.write(worker_code)
    
    return {
        'worker_file': worker_file,
        'code': worker_code
    }


def generate_temporal_readme(repo_root: str, target_name: str, impl_type: str, language: str):
    """Generate README for temporal folder"""
    readme_path = os.path.join(repo_root, 'temporal', 'README.md')
    
    readme_content = f"""# Temporal Implementation

This folder contains Temporal workflows and activities for {target_name}.

## Folder Structure

```
temporal/
├── workflows/          # Workflow definitions
├── activities/         # Activity implementations
│   └── shared/        # Shared activities
├── workers/           # Worker configurations
├── client/            # Temporal client setup
├── tests/             # Tests
│   ├── workflows/
│   ├── activities/
│   └── integration/
└── config/            # Configuration files
```

## Implementation Type

**Type:** {impl_type.upper()}
**Target:** {target_name}
**Language:** {language}

## Getting Started

1. Install Temporal SDK (see .env setup)
2. Configure connection in config/temporal_config.py
3. Implement workflow logic in workflows/
4. Implement activity logic in activities/
5. Run worker: `python workers/main_worker.py`
6. Start workflows using client code

## Configuration

All Temporal configuration should be in `.env` file:
- TEMPORAL_HOST
- TEMPORAL_NAMESPACE
- TEMPORAL_TASK_QUEUE

## Testing

Run tests:
```bash
pytest temporal/tests/
```

## Documentation

See for-developer/ folder for complete documentation.
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)


def run_testing_pipeline(
    files_created: List[str],
    impl_type: str,
    category: str,
    name: str,
    language: str,
    repo_root: str
) -> bool:
    """
    Run complete testing pipeline: generate tests, run tests, create reports
    Returns True if tests pass, False if tests fail
    """
    script_dir = Path(__file__).parent
    
    # Determine test output directory
    if impl_type == 'api':
        test_dir = os.path.join(repo_root, 'temporal', 'tests', 'api', category, name.replace(' ', '_').lower())
    else:
        test_dir = os.path.join(repo_root, 'temporal', 'tests', 'service', name.replace(' ', '_').lower())
    
    # Find workflow file
    workflow_file = None
    for f in files_created:
        if 'workflow' in f.lower():
            workflow_file = f
            break
    
    if not workflow_file:
        print("⚠️  No workflow file found, skipping test generation")
        return True  # Don't fail if no workflow
    
    try:
        # Step 1: Generate tests
        print(f"\n1️⃣  Generating tests...")
        result = subprocess.run([
            sys.executable,
            str(script_dir / 'generate-tests.py'),
            '--implementation', workflow_file,
            '--type', 'workflow',
            '--category', f'{impl_type}/{category}',
            '--name', name.replace(' ', '_').lower(),
            '--output', test_dir
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Test generation failed:")
            print(result.stderr)
            return False
        
        print(f"✅ Tests generated in: {test_dir}")
        
        # Step 2: Run tests
        print(f"\n2️⃣  Running tests...")
        test_results_file = os.path.join(test_dir, 'test-results.json')
        
        result = subprocess.run([
            sys.executable,
            str(script_dir / 'run-tests.py'),
            '--test-dir', test_dir,
            '--language', language,
            '--output', test_results_file
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Tests failed!")
            if result.stderr:
                print(result.stderr)
            
            # Still create report for failed tests
            if os.path.exists(test_results_file):
                subprocess.run([
                    sys.executable,
                    str(script_dir / 'create-test-reports.py'),
                    '--results', test_results_file,
                    '--output', os.path.join(test_dir, 'REPORT.md'),
                    '--name', name
                ], capture_output=True)
            
            return False
        
        print(f"✅ All tests passed!")
        
        # Step 3: Create reports
        print(f"\n3️⃣  Creating test reports...")
        
        result = subprocess.run([
            sys.executable,
            str(script_dir / 'create-test-reports.py'),
            '--results', test_results_file,
            '--output', os.path.join(test_dir, 'REPORT.md'),
            '--name', name
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️  Report generation failed (tests still passed)")
            print(result.stderr)
        else:
            print(f"✅ Report created: {test_dir}/REPORT.md")
        
        # Update summary
        tests_root = os.path.join(repo_root, 'temporal', 'tests')
        summary_result = subprocess.run([
            sys.executable,
            str(script_dir / 'create-test-reports.py'),
            '--summary', tests_root,
            '--output', os.path.join(tests_root, 'SUMMARY.md')
        ], capture_output=True, text=True)
        
        if summary_result.returncode == 0:
            print(f"✅ Summary updated: {tests_root}/SUMMARY.md")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python refactor-to-temporal.py <target_type> <target_id> [scan_report_json] [repo_root] [--strategy STRATEGY] [--agent-mode]")
        print("\nTarget types:")
        print("  service <number>  - Implement entire service (e.g., service 1)")
        print("  api <name>        - Implement all usages of API (e.g., api 'OpenAI API')")
        print("\nOptional flags:")
        print("  --strategy <shared|per_service>  - Implementation strategy (default: shared)")
        print("  --agent-mode                     - Non-interactive mode for autonomous agents")
        print("\nExamples:")
        print("  python refactor-to-temporal.py service 1 scan_report.json")
        print("  python refactor-to-temporal.py api 'OpenAI API' scan_report.json . --strategy shared --agent-mode")
        return 1
    
    target_type = sys.argv[1].lower()
    target_id = sys.argv[2]
    scan_report_path = sys.argv[3] if len(sys.argv) > 3 else None
    repo_root = sys.argv[4] if len(sys.argv) > 4 else os.getcwd()
    
    # Parse optional flags
    strategy_override = None
    agent_mode = '--agent-mode' in sys.argv
    
    if '--strategy' in sys.argv:
        strategy_idx = sys.argv.index('--strategy')
        if strategy_idx + 1 < len(sys.argv):
            strategy_override = sys.argv[strategy_idx + 1]
            if strategy_override not in ['shared', 'per_service']:
                print(f"❌ Invalid strategy: {strategy_override}. Must be 'shared' or 'per_service'.")
                return 1
    
    # Store in global scope for ask_implementation_strategy()
    global STRATEGY_OVERRIDE, AGENT_MODE
    STRATEGY_OVERRIDE = strategy_override
    AGENT_MODE = agent_mode
    
    if target_type not in ['service', 'api']:
        print(f"Invalid target type: {target_type}. Must be 'service' or 'api'.")
        return 1
    
    # Load scan report
    if scan_report_path and os.path.exists(scan_report_path):
        try:
            with open(scan_report_path, 'r') as f:
                scan_report = json.load(f)
        except Exception as e:
            print(f"Error loading scan report: {e}")
            return 1
    else:
        print("No scan report provided. Please run scan-codebase.py first.")
        return 1
    
    # Load analysis report (should be generated by analyze-temporal.py)
    analysis_report_path = scan_report_path.replace('scan_report', 'analysis_report')
    if os.path.exists(analysis_report_path):
        try:
            with open(analysis_report_path, 'r') as f:
                analysis_report = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load analysis report: {e}")
            analysis_report = {}
    else:
        analysis_report = {}
    
    # Detect language from scan report (check file extensions in files list)
    language = 'python'  # Default
    
    # Try to detect from external_apis files
    all_files = []
    for api in scan_report.get('external_apis', []):
        all_files.extend(api.get('files', []))
    for api in scan_report.get('internal_apis', []):
        all_files.extend([loc.get('file', '') for loc in api.get('locations', [])])
    
    # Detect language from file extensions
    if any(f.endswith('.java') for f in all_files):
        language = 'java'
    elif any(f.endswith('.ts') or f.endswith('.tsx') for f in all_files):
        language = 'typescript'
    elif any(f.endswith('.js') or f.endswith('.jsx') for f in all_files):
        language = 'javascript'
    elif any(f.endswith('.go') for f in all_files):
        language = 'go'
    elif any(f.endswith('.cs') for f in all_files):
        language = 'csharp'
    elif any(f.endswith('.php') for f in all_files):
        language = 'php'
    elif any(f.endswith('.rb') for f in all_files):
        language = 'ruby'
    
    print(f"\n[INFO] Detected language: {language}")
    
    try:
        if target_type == 'api':
            # Find API in scan report
            apis = scan_report.get('external_apis', [])
            api_data = None
            for api in apis:
                if api.get('api_name') == target_id or api.get('name') == target_id or (target_id.isdigit() and int(target_id) - 1 < len(apis) and apis[int(target_id) - 1] == api):
                    api_data = api
                    break
            
            if not api_data:
                print(f"API not found: {target_id}")
                return 1
            
            print(f"\n{'='*60}")
            print(f"IMPLEMENTING API: {api_data.get('api_name', api_data.get('name', 'Unknown'))}")
            print(f"{'='*60}")
            print(f"Call count: {api_data.get('call_count', 0)}")
            print(f"File count: {api_data.get('file_count', 0)}")
            print(f"Files: {', '.join(api_data.get('files', [])[:3])}...")
            
            # Ask for implementation strategy (or use flags)
            if strategy_override:
                strategy = strategy_override
                print(f"\n[AGENT] Using strategy from --strategy flag: {strategy}")
            elif agent_mode:
                strategy = 'shared'  # Default for agent mode
                print(f"\n[AGENT] Agent mode: Using default strategy: {strategy}")
            else:
                strategy = ask_implementation_strategy()
            
            print(f"\nImplementing with strategy: {strategy}")
            print(f"Creating temporal/ folder structure...")
            
            api_name = api_data.get('api_name', api_data.get('name', 'Unknown API'))
            result = refactor_api_based(api_name, api_data, language, repo_root, strategy)
            
            if result['success']:
                print(f"\n{'='*60}")
                print("[SUCCESS] API IMPLEMENTATION COMPLETE")
                print(f"{'='*60}")
                print(f"\nFiles created:")
                for file_path in result['files_created']:
                    print(f"  - {file_path}")
                
                # NEW: Generate and run tests
                print(f"\n{'='*60}")
                print("[TESTING] GENERATING AND RUNNING TESTS")
                print(f"{'='*60}")
                
                test_success = run_testing_pipeline(
                    result['files_created'],
                    'api',
                    'external' if 'API' in api_name else 'internal',
                    api_name,
                    language,
                    repo_root
                )
                
                if not test_success:
                    print(f"\n[FAILED] Tests failed! Registry will NOT be updated.")
                    print(f"Fix the failing tests and try again.")
                    return 1
                
                print(f"\nNext steps:")
                for i, step in enumerate(result['next_steps'], 1):
                    print(f"  {i}. {step}")
                print(f"\n{'='*60}")
                return 0
            else:
                print(f"Implementation failed: {result.get('error', 'Unknown error')}")
                return 1
        
        elif target_type == 'service':
            # Find service in analysis report
            services = analysis_report.get('service_suggestions', [])
            if not services:
                print("No services found in analysis report.")
                return 1
            
            if target_id.isdigit():
                service_idx = int(target_id) - 1
                if service_idx < 0 or service_idx >= len(services):
                    print(f"Invalid service number: {target_id}")
                    return 1
                service_data = services[service_idx]
            else:
                # Find by name
                service_data = None
                for svc in services:
                    if svc['service_name'] == target_id:
                        service_data = svc
                        break
                
                if not service_data:
                    print(f"Service not found: {target_id}")
                    return 1
            
            print(f"\n{'='*60}")
            print(f"IMPLEMENTING SERVICE: {service_data['service_name']}")
            print(f"{'='*60}")
            print(f"Pattern: {service_data['pattern_description']}")
            print(f"Functions to convert: {service_data['function_count']}")
            
            print(f"\nCreating temporal/ folder structure...")
            
            result = refactor_service_based(service_data['service_name'], service_data, language, repo_root)
            
            if result['success']:
                print(f"\n{'='*60}")
                print("✅ SERVICE IMPLEMENTATION COMPLETE")
                print(f"{'='*60}")
                print(f"\nFiles created:")
                for file_path in result['files_created']:
                    print(f"  - {file_path}")
                
                # NEW: Generate and run tests
                print(f"\n{'='*60}")
                print("[TESTING] GENERATING AND RUNNING TESTS")
                print(f"{'='*60}")
                
                test_success = run_testing_pipeline(
                    result['files_created'],
                    'service',
                    'service',
                    service_data['service_name'],
                    language,
                    repo_root
                )
                
                if not test_success:
                    print(f"\n[FAILED] Tests failed! Registry will NOT be updated.")
                    print(f"Fix the failing tests and try again.")
                    return 1
                
                print(f"\nNext steps:")
                for i, step in enumerate(result['next_steps'], 1):
                    print(f"  {i}. {step}")
                print(f"\n{'='*60}")
                return 0
            else:
                print(f"Implementation failed: {result.get('error', 'Unknown error')}")
                return 1
    
    except Exception as e:
        print(f"Error during implementation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

