#!/usr/bin/env python3
"""
Temporal Integration System - Test Generator
Generates unit tests for Temporal implementations (workflows, activities, workers)

Usage:
    python generate-tests.py --implementation <file> --type <workflow|activity|worker> \
                             --category <api/internal|api/external|service> \
                             --name <api_or_service_name> \
                             --output <test_directory>
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional


# Test templates for each language
TEST_TEMPLATES = {
    'python': {
        'workflow': '''import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from {workflow_module} import {workflow_class}
from activities import {activity_imports}


@pytest.mark.asyncio
async def test_{workflow_name}_happy_path():
    """Test successful workflow execution"""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[{workflow_class}],
            activities=[{activity_list}]
        ):
            # Execute workflow
            result = await env.client.execute_workflow(
                {workflow_class}.run,
                # TODO: Add test input data
                id="test-workflow-1",
                task_queue="test-queue",
            )
            
            # Assert expected result
            assert result is not None
            # TODO: Add specific assertions


@pytest.mark.asyncio
async def test_{workflow_name}_with_invalid_input():
    """Test workflow with invalid input"""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[{workflow_class}],
            activities=[{activity_list}]
        ):
            # Execute workflow with invalid input
            with pytest.raises(Exception):
                await env.client.execute_workflow(
                    {workflow_class}.run,
                    # TODO: Add invalid input data
                    id="test-workflow-2",
                    task_queue="test-queue",
                )


@pytest.mark.asyncio
async def test_{workflow_name}_activity_failure():
    """Test workflow handles activity failure"""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[{workflow_class}],
            activities=[{activity_list}]
        ):
            # TODO: Mock activity to fail
            # Execute workflow
            result = await env.client.execute_workflow(
                {workflow_class}.run,
                # TODO: Add test input
                id="test-workflow-3",
                task_queue="test-queue",
            )
            
            # Verify retry behavior
            # TODO: Add assertions for retry logic


@pytest.mark.asyncio
async def test_{workflow_name}_timeout():
    """Test workflow timeout handling"""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[{workflow_class}],
            activities=[{activity_list}]
        ):
            # Test timeout scenario
            with pytest.raises(TimeoutError):
                await env.client.execute_workflow(
                    {workflow_class}.run,
                    # TODO: Add input that causes timeout
                    id="test-workflow-4",
                    task_queue="test-queue",
                    execution_timeout="1s"
                )
''',
        'activity': '''import pytest
from activities.{activity_module} import {activity_functions}


@pytest.mark.asyncio
async def test_{activity_name}_success():
    """Test activity executes successfully"""
    # TODO: Add test input
    result = await {activity_function}(test_input)
    
    # Assert expected behavior
    assert result is not None
    # TODO: Add specific assertions


@pytest.mark.asyncio
async def test_{activity_name}_with_invalid_input():
    """Test activity handles invalid input"""
    with pytest.raises(ValueError):
        await {activity_function}(invalid_input)


@pytest.mark.asyncio
async def test_{activity_name}_retry_on_failure():
    """Test activity retry behavior"""
    # TODO: Mock external dependency to fail
    # Verify retry logic works
    pass


@pytest.mark.asyncio
async def test_{activity_name}_timeout():
    """Test activity timeout handling"""
    # TODO: Mock slow operation
    # Verify timeout is enforced
    pass
''',
        'integration': '''import pytest
from temporalio.client import Client
from temporalio.worker import Worker
from {workflow_module} import {workflow_class}
from activities import {activity_imports}


@pytest.mark.asyncio
async def test_full_workflow_integration():
    """Test complete workflow with real activities"""
    # Connect to test server
    client = await Client.connect("localhost:7233")
    
    async with Worker(
        client,
        task_queue="test-queue",
        workflows=[{workflow_class}],
        activities=[{activity_list}]
    ):
        # Execute full workflow
        result = await client.execute_workflow(
            {workflow_class}.run,
            # TODO: Add test input
            id="integration-test-1",
            task_queue="test-queue",
        )
        
        # Verify end-to-end behavior
        assert result is not None
        # TODO: Add integration assertions


@pytest.mark.asyncio
async def test_workflow_with_real_external_api():
    """Test workflow with actual external API calls"""
    # TODO: Set up test API credentials
    # Execute workflow
    # Verify results
    pass
'''
    },
    'typescript': {
        'workflow': '''import {{ describe, it, expect }} from '@jest/globals';
import {{ TestWorkflowEnvironment }} from '@temporalio/testing';
import {{ Worker }} from '@temporalio/worker';
import {{ {workflow_name} }} from '../workflows/{workflow_file}';
import * as activities from '../activities';


describe('{workflow_name} Workflow', () => {{
  let testEnv: TestWorkflowEnvironment;

  beforeAll(async () => {{
    testEnv = await TestWorkflowEnvironment.createTimeSkipping();
  }});

  afterAll(async () => {{
    await testEnv?.teardown();
  }});

  it('should execute successfully with valid input', async () => {{
    const {{ client }} = testEnv;
    const worker = await Worker.create({{
      connection: testEnv.nativeConnection,
      taskQueue: 'test-queue',
      workflowsPath: require.resolve('../workflows/{workflow_file}'),
      activities,
    }});

    await worker.runUntil(async () => {{
      // TODO: Add test input
      const result = await client.workflow.execute({workflow_name}, {{
        taskQueue: 'test-queue',
        workflowId: 'test-workflow-1',
        args: [/* TODO: test args */],
      }});

      // TODO: Add assertions
      expect(result).toBeDefined();
    }});
  }});

  it('should handle invalid input', async () => {{
    const {{ client }} = testEnv;
    const worker = await Worker.create({{
      connection: testEnv.nativeConnection,
      taskQueue: 'test-queue',
      workflowsPath: require.resolve('../workflows/{workflow_file}'),
      activities,
    }});

    await worker.runUntil(async () => {{
      // TODO: Test with invalid input
      await expect(
        client.workflow.execute({workflow_name}, {{
          taskQueue: 'test-queue',
          workflowId: 'test-workflow-2',
          args: [/* invalid args */],
        }})
      ).rejects.toThrow();
    }});
  }});

  it('should handle activity failures', async () => {{
    const {{ client }} = testEnv;
    // TODO: Mock activities to fail
    const worker = await Worker.create({{
      connection: testEnv.nativeConnection,
      taskQueue: 'test-queue',
      workflowsPath: require.resolve('../workflows/{workflow_file}'),
      activities,
    }});

    await worker.runUntil(async () => {{
      // TODO: Test retry behavior
    }});
  }});

  it('should handle timeouts', async () => {{
    const {{ client }} = testEnv;
    const worker = await Worker.create({{
      connection: testEnv.nativeConnection,
      taskQueue: 'test-queue',
      workflowsPath: require.resolve('../workflows/{workflow_file}'),
      activities,
    }});

    await worker.runUntil(async () => {{
      // TODO: Test timeout scenario
    }});
  }});
}});
''',
        'activity': '''import {{ describe, it, expect }} from '@jest/globals';
import {{ {activity_functions} }} from '../activities/{activity_file}';


describe('{activity_name} Activity', () => {{
  it('should execute successfully', async () => {{
    // TODO: Add test input
    const result = await {activity_function}(testInput);
    
    // TODO: Add assertions
    expect(result).toBeDefined();
  }});

  it('should handle invalid input', async () => {{
    // TODO: Test with invalid input
    await expect({activity_function}(invalidInput)).rejects.toThrow();
  }});

  it('should retry on failure', async () => {{
    // TODO: Mock external dependency to fail
    // Verify retry behavior
  }});

  it('should handle timeouts', async () => {{
    // TODO: Mock slow operation
    // Verify timeout handling
  }});
}});
''',
        'integration': '''import {{ describe, it, expect }} from '@jest/globals';
import {{ Client }} from '@temporalio/client';
import {{ Worker }} from '@temporalio/worker';
import {{ {workflow_name} }} from '../workflows/{workflow_file}';
import * as activities from '../activities';


describe('{workflow_name} Integration Tests', () => {{
  let client: Client;

  beforeAll(async () => {{
    client = await Client.connect({{ address: 'localhost:7233' }});
  }});

  it('should complete full workflow with real activities', async () => {{
    const worker = await Worker.create({{
      connection: await client.connection.connect(),
      taskQueue: 'test-queue',
      workflowsPath: require.resolve('../workflows/{workflow_file}'),
      activities,
    }});

    await worker.runUntil(async () => {{
      // TODO: Execute full workflow
      const result = await client.workflow.execute({workflow_name}, {{
        taskQueue: 'test-queue',
        workflowId: 'integration-test-1',
        args: [/* TODO: test args */],
      }});

      // TODO: Verify end-to-end behavior
      expect(result).toBeDefined();
    }});
  }});

  it('should work with real external APIs', async () => {{
    // TODO: Set up test API credentials
    // Execute workflow with real API calls
    // Verify results
  }});
}});
'''
    },
    'go': {
        'workflow': '''package workflows_test

import (
\t"testing"
\t"time"

\t"github.com/stretchr/testify/suite"
\t"go.temporal.io/sdk/testsuite"
\t"go.temporal.io/sdk/worker"

\t"your-module/temporal/workflows"
\t"your-module/temporal/activities"
)

type {WorkflowClass}TestSuite struct {{
\tsuite.Suite
\ttestsuite.WorkflowTestSuite
}}

func Test{WorkflowClass}(t *testing.T) {{
\tsuite.Run(t, new({WorkflowClass}TestSuite))
}}

func (s *{WorkflowClass}TestSuite) Test_{workflow_name}_Success() {{
\tenv := s.NewTestWorkflowEnvironment()
\t
\t// TODO: Register activities
\tenv.RegisterActivity(activities.{ActivityName})
\t
\t// TODO: Execute workflow
\tenv.ExecuteWorkflow(workflows.{WorkflowClass})
\t
\t// TODO: Assert success
\ts.True(env.IsWorkflowCompleted())
\ts.NoError(env.GetWorkflowError())
}}

func (s *{WorkflowClass}TestSuite) Test_{workflow_name}_WithInvalidInput() {{
\tenv := s.NewTestWorkflowEnvironment()
\t
\t// TODO: Test with invalid input
\tenv.ExecuteWorkflow(workflows.{WorkflowClass})
\t
\t// TODO: Assert error
\ts.Error(env.GetWorkflowError())
}}

func (s *{WorkflowClass}TestSuite) Test_{workflow_name}_ActivityFailure() {{
\tenv := s.NewTestWorkflowEnvironment()
\t
\t// TODO: Mock activity to fail
\tenv.OnActivity(activities.{ActivityName}, nil).Return(nil, errors.New("test error"))
\t
\t// Execute workflow
\tenv.ExecuteWorkflow(workflows.{WorkflowClass})
\t
\t// TODO: Verify retry behavior
}}

func (s *{WorkflowClass}TestSuite) Test_{workflow_name}_Timeout() {{
\tenv := s.NewTestWorkflowEnvironment()
\tenv.SetWorkflowTimeout(1 * time.Second)
\t
\t// TODO: Test timeout scenario
}}
''',
        'activity': '''package activities_test

import (
\t"testing"

\t"github.com/stretchr/testify/assert"
\t"your-module/temporal/activities"
)

func Test_{ActivityName}_Success(t *testing.T) {{
\t// TODO: Add test input
\tresult, err := activities.{ActivityName}(context.Background(), testInput)
\t
\t// TODO: Add assertions
\tassert.NoError(t, err)
\tassert.NotNil(t, result)
}}

func Test_{ActivityName}_WithInvalidInput(t *testing.T) {{
\t// TODO: Test with invalid input
\tresult, err := activities.{ActivityName}(context.Background(), invalidInput)
\t
\tassert.Error(t, err)
}}

func Test_{ActivityName}_RetryOnFailure(t *testing.T) {{
\t// TODO: Mock external dependency to fail
\t// Verify retry behavior
}}

func Test_{ActivityName}_Timeout(t *testing.T) {{
\t// TODO: Mock slow operation
\t// Verify timeout handling
}}
''',
        'integration': '''package integration_test

import (
\t"testing"
\t"time"

\t"go.temporal.io/sdk/client"
\t"go.temporal.io/sdk/worker"

\t"your-module/temporal/workflows"
\t"your-module/temporal/activities"
)

func TestFullWorkflowIntegration(t *testing.T) {{
\t// Connect to test server
\tc, err := client.Dial(client.Options{{
\t\tHostPort: "localhost:7233",
\t}})
\tif err != nil {{
\t\tt.Fatal(err)
\t}}
\tdefer c.Close()
\t
\t// Create worker
\tw := worker.New(c, "test-queue", worker.Options{{}})
\tw.RegisterWorkflow(workflows.{WorkflowClass})
\tw.RegisterActivity(activities.{ActivityName})
\t
\tgo func() {{
\t\tif err := w.Run(worker.InterruptCh()); err != nil {{
\t\t\tt.Error(err)
\t\t}}
\t}}()
\tdefer w.Stop()
\t
\t// TODO: Execute workflow
\t// TODO: Verify end-to-end behavior
}}
'''
    },
    'java': {
        'workflow': '''package com.example.temporal.workflows;

import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

class {WorkflowClass}Test {{

    private TestWorkflowEnvironment testEnv;
    private Worker worker;
    private WorkflowClient client;

    @BeforeEach
    void setUp() {{
        testEnv = TestWorkflowEnvironment.newInstance();
        worker = testEnv.newWorker("test-queue");
        worker.registerWorkflowImplementationTypes({WorkflowClass}Impl.class);
        // TODO: Register activities
        client = testEnv.getWorkflowClient();
    }}

    @AfterEach
    void tearDown() {{
        testEnv.close();
    }}

    @Test
    void test{WorkflowClass}_Success() {{
        testEnv.start();
        
        // TODO: Create workflow stub
        {WorkflowClass} workflow = client.newWorkflowStub(
            {WorkflowClass}.class,
            WorkflowOptions.newBuilder()
                .setTaskQueue("test-queue")
                .setWorkflowId("test-workflow-1")
                .build()
        );
        
        // TODO: Execute workflow
        // TODO: Add assertions
    }}

    @Test
    void test{WorkflowClass}_WithInvalidInput() {{
        testEnv.start();
        
        // TODO: Test with invalid input
        // TODO: Assert exception is thrown
    }}

    @Test
    void test{WorkflowClass}_ActivityFailure() {{
        testEnv.start();
        
        // TODO: Mock activity to fail
        // TODO: Verify retry behavior
    }}

    @Test
    void test{WorkflowClass}_Timeout() {{
        testEnv.start();
        
        // TODO: Test timeout scenario
    }}
}}
''',
        'activity': '''package com.example.temporal.activities;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class {ActivityClass}Test {{

    @Test
    void test{ActivityName}_Success() {{
        // TODO: Create activity instance
        {ActivityClass} activity = new {ActivityClass}Impl();
        
        // TODO: Add test input
        // TODO: Execute activity
        // TODO: Add assertions
    }}

    @Test
    void test{ActivityName}_WithInvalidInput() {{
        // TODO: Test with invalid input
        // TODO: Assert exception
    }}

    @Test
    void test{ActivityName}_RetryOnFailure() {{
        // TODO: Mock external dependency to fail
        // Verify retry behavior
    }}

    @Test
    void test{ActivityName}_Timeout() {{
        // TODO: Mock slow operation
        // Verify timeout handling
    }}
}}
''',
        'integration': '''package com.example.temporal.integration;

import io.temporal.client.WorkflowClient;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import org.junit.jupiter.api.*;

class {WorkflowClass}IntegrationTest {{

    private WorkflowServiceStubs service;
    private WorkflowClient client;
    private WorkerFactory factory;

    @BeforeEach
    void setUp() {{
        service = WorkflowServiceStubs.newLocalServiceStubs();
        client = WorkflowClient.newInstance(service);
        factory = WorkerFactory.newInstance(client);
    }}

    @AfterEach
    void tearDown() {{
        factory.shutdown();
        service.shutdown();
    }}

    @Test
    void testFullWorkflowIntegration() {{
        // TODO: Register workflows and activities
        Worker worker = factory.newWorker("test-queue");
        // worker.registerWorkflowImplementationTypes(...);
        // worker.registerActivitiesImplementations(...);
        
        factory.start();
        
        // TODO: Execute full workflow
        // TODO: Verify end-to-end behavior
    }}
}}
'''
    },
    'csharp': {
        'workflow': '''using Xunit;
using Temporalio.Testing;
using Temporalio.Worker;

namespace TemporalWorkflows.Tests
{{
    public class {WorkflowClass}Tests
    {{
        [Fact]
        public async Task Test_{workflow_name}_Success()
        {{
            await using var env = await WorkflowEnvironment.StartTimeSkippingAsync();
            
            using var worker = new TemporalWorker(
                env.Client,
                new TemporalWorkerOptions("test-queue")
                    .AddWorkflow<{WorkflowClass}>()
                    .AddAllActivities(new {ActivityClass}())
            );
            await worker.ExecuteAsync(async () =>
            {{
                // TODO: Execute workflow
                // TODO: Add assertions
            }});
        }}

        [Fact]
        public async Task Test_{workflow_name}_WithInvalidInput()
        {{
            // TODO: Test with invalid input
        }}

        [Fact]
        public async Task Test_{workflow_name}_ActivityFailure()
        {{
            // TODO: Mock activity to fail
            // TODO: Verify retry behavior
        }}

        [Fact]
        public async Task Test_{workflow_name}_Timeout()
        {{
            // TODO: Test timeout scenario
        }}
    }}
}}
''',
        'activity': '''using Xunit;

namespace TemporalActivities.Tests
{{
    public class {ActivityClass}Tests
    {{
        [Fact]
        public async Task Test_{ActivityName}_Success()
        {{
            var activity = new {ActivityClass}();
            
            // TODO: Add test input
            // TODO: Execute activity
            // TODO: Add assertions
        }}

        [Fact]
        public async Task Test_{ActivityName}_WithInvalidInput()
        {{
            // TODO: Test with invalid input
        }}

        [Fact]
        public async Task Test_{ActivityName}_RetryOnFailure()
        {{
            // TODO: Mock external dependency to fail
        }}

        [Fact]
        public async Task Test_{ActivityName}_Timeout()
        {{
            // TODO: Mock slow operation
        }}
    }}
}}
''',
        'integration': '''using Xunit;
using Temporalio.Client;
using Temporalio.Worker;

namespace TemporalIntegration.Tests
{{
    public class {WorkflowClass}IntegrationTests
    {{
        [Fact]
        public async Task TestFullWorkflowIntegration()
        {{
            var client = await TemporalClient.ConnectAsync(new("localhost:7233"));
            
            using var worker = new TemporalWorker(
                client,
                new TemporalWorkerOptions("test-queue")
                    .AddWorkflow<{WorkflowClass}>()
                    .AddAllActivities(new {ActivityClass}())
            );
            
            await worker.ExecuteAsync(async () =>
            {{
                // TODO: Execute full workflow
                // TODO: Verify end-to-end behavior
            }});
        }}
    }}
}}
'''
    },
    'php': {
        'workflow': '''<?php

declare(strict_types=1);

namespace Tests\\Workflows;

use PHPUnit\\Framework\\TestCase;
use Temporal\\Testing\\WorkflowTestCase;

final class {WorkflowClass}Test extends WorkflowTestCase
{{
    public function test_{workflow_name}_Success(): void
    {{
        // TODO: Execute workflow
        // TODO: Add assertions
    }}

    public function test_{workflow_name}_WithInvalidInput(): void
    {{
        $this->expectException(\\InvalidArgumentException::class);
        // TODO: Test with invalid input
    }}

    public function test_{workflow_name}_ActivityFailure(): void
    {{
        // TODO: Mock activity to fail
        // TODO: Verify retry behavior
    }}

    public function test_{workflow_name}_Timeout(): void
    {{
        // TODO: Test timeout scenario
    }}
}}
''',
        'activity': '''<?php

declare(strict_types=1);

namespace Tests\\Activities;

use PHPUnit\\Framework\\TestCase;

final class {ActivityClass}Test extends TestCase
{{
    public function test_{ActivityName}_Success(): void
    {{
        // TODO: Create activity instance
        // TODO: Add test input
        // TODO: Execute activity
        // TODO: Add assertions
    }}

    public function test_{ActivityName}_WithInvalidInput(): void
    {{
        $this->expectException(\\InvalidArgumentException::class);
        // TODO: Test with invalid input
    }}

    public function test_{ActivityName}_RetryOnFailure(): void
    {{
        // TODO: Mock external dependency to fail
    }}

    public function test_{ActivityName}_Timeout(): void
    {{
        // TODO: Mock slow operation
    }}
}}
''',
        'integration': '''<?php

declare(strict_types=1);

namespace Tests\\Integration;

use PHPUnit\\Framework\\TestCase;
use Temporal\\Client\\WorkflowClient;

final class {WorkflowClass}IntegrationTest extends TestCase
{{
    public function testFullWorkflowIntegration(): void
    {{
        // TODO: Connect to Temporal server
        // TODO: Execute full workflow
        // TODO: Verify end-to-end behavior
    }}
}}
'''
    },
    'ruby': {
        'workflow': '''require 'spec_helper'
require 'temporal/testing'

RSpec.describe {WorkflowClass} do
  let(:env) {{ Temporal::Testing::WorkflowEnvironment.new }}

  describe '#{workflow_name}' do
    it 'executes successfully with valid input' do
      # TODO: Execute workflow
      # TODO: Add expectations
    end

    it 'handles invalid input' do
      expect {{
        # TODO: Execute with invalid input
      }}.to raise_error(ArgumentError)
    end

    it 'handles activity failures' do
      # TODO: Mock activity to fail
      # TODO: Verify retry behavior
    end

    it 'handles timeouts' do
      # TODO: Test timeout scenario
    end
  end
end
''',
        'activity': '''require 'spec_helper'

RSpec.describe {ActivityClass} do
  describe '#{activity_name}' do
    it 'executes successfully' do
      # TODO: Create activity instance
      # TODO: Add test input
      # TODO: Execute activity
      # TODO: Add expectations
    end

    it 'handles invalid input' do
      expect {{
        # TODO: Execute with invalid input
      }}.to raise_error(ArgumentError)
    end

    it 'retries on failure' do
      # TODO: Mock external dependency to fail
    end

    it 'handles timeouts' do
      # TODO: Mock slow operation
    end
  end
end
''',
        'integration': '''require 'spec_helper'
require 'temporal/client'

RSpec.describe '{WorkflowClass} Integration' do
  let(:client) {{ Temporal::Client.generate(host: 'localhost:7233') }}

  it 'completes full workflow with real activities' do
    # TODO: Execute full workflow
    # TODO: Verify end-to-end behavior
  end
end
'''
    }
}


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension"""
    ext = Path(file_path).suffix.lower()
    
    ext_to_lang = {
        '.py': 'python',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.js': 'typescript',  # Use TypeScript templates for JS
        '.jsx': 'typescript',
        '.go': 'go',
        '.java': 'java',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby'
    }
    
    return ext_to_lang.get(ext)


def extract_workflow_info(file_path: str, language: str) -> Dict[str, str]:
    """Extract workflow class/function name from implementation file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return {}
    
    info = {}
    
    if language == 'python':
        # Look for workflow class
        class_match = re.search(r'class\s+(\w+).*?@workflow\.defn', content, re.DOTALL)
        if class_match:
            info['workflow_class'] = class_match.group(1)
            info['workflow_name'] = class_match.group(1).lower().replace('workflow', '')
        
        # Look for workflow module
        module_name = Path(file_path).stem
        info['workflow_module'] = f'workflows.{module_name}'
        
    elif language == 'typescript':
        # Look for workflow function
        func_match = re.search(r'export\s+(?:async\s+)?function\s+(\w+)', content)
        if func_match:
            info['workflow_name'] = func_match.group(1)
            info['workflow_file'] = Path(file_path).stem
    
    elif language == 'go':
        # Look for workflow function
        func_match = re.search(r'func\s+(\w+)\s*\(', content)
        if func_match:
            info['WorkflowClass'] = func_match.group(1)
            info['workflow_name'] = func_match.group(1).lower()
    
    elif language == 'java':
        # Look for workflow class
        class_match = re.search(r'class\s+(\w+)', content)
        if class_match:
            info['WorkflowClass'] = class_match.group(1)
    
    elif language == 'csharp':
        # Look for workflow class
        class_match = re.search(r'class\s+(\w+)', content)
        if class_match:
            info['WorkflowClass'] = class_match.group(1)
            info['workflow_name'] = class_match.group(1).lower()
    
    elif language == 'php':
        # Look for workflow class
        class_match = re.search(r'class\s+(\w+)', content)
        if class_match:
            info['WorkflowClass'] = class_match.group(1)
            info['workflow_name'] = class_match.group(1).lower()
    
    elif language == 'ruby':
        # Look for workflow class
        class_match = re.search(r'class\s+(\w+)', content)
        if class_match:
            info['WorkflowClass'] = class_match.group(1)
            info['workflow_name'] = class_match.group(1).downcase
    
    return info


def generate_test_file(
    impl_file: str,
    test_type: str,
    language: str,
    output_dir: str,
    api_name: str
) -> Optional[str]:
    """Generate a test file for the implementation"""
    
    # Get template
    if language not in TEST_TEMPLATES:
        print(f"Warning: No test template for language: {language}")
        return None
    
    if test_type not in TEST_TEMPLATES[language]:
        print(f"Warning: No {test_type} template for {language}")
        return None
    
    template = TEST_TEMPLATES[language][test_type]
    
    # Extract info from implementation
    info = extract_workflow_info(impl_file, language)
    
    # Add defaults
    info.setdefault('workflow_name', api_name.replace('-', '_'))
    info.setdefault('workflow_class', api_name.replace('-', '_').title() + 'Workflow')
    info.setdefault('activity_imports', '# TODO: Add activity imports')
    info.setdefault('activity_list', '# TODO: Add activities')
    info.setdefault('activity_name', api_name.replace('-', '_'))
    info.setdefault('activity_function', api_name.replace('-', '_') + '_activity')
    info.setdefault('activity_functions', api_name.replace('-', '_') + '_activity')
    info.setdefault('activity_module', api_name.replace('-', '_'))
    info.setdefault('activity_file', api_name.replace('-', '_'))
    info.setdefault('WorkflowClass', api_name.replace('-', '_').title() + 'Workflow')
    info.setdefault('ActivityName', api_name.replace('-', '_').title() + 'Activity')
    info.setdefault('ActivityClass', api_name.replace('-', '_').title() + 'Activity')
    info.setdefault('workflow_file', api_name.replace('-', '_'))
    
    # Format template
    try:
        test_content = template.format(**info)
    except KeyError as e:
        print(f"Warning: Missing template variable: {e}")
        return None
    
    # Determine output filename
    file_extensions = {
        'python': '.py',
        'typescript': '.test.ts',
        'go': '_test.go',
        'java': '.java',
        'csharp': '.cs',
        'php': '.php',
        'ruby': '_spec.rb'
    }
    
    ext = file_extensions.get(language, '.txt')
    
    test_filename_patterns = {
        'python': f'test_{test_type}.py',
        'typescript': f'{test_type}.test.ts',
        'go': f'{test_type}_test.go',
        'java': f'{info.get("WorkflowClass", "Test")}Test.java',
        'csharp': f'{info.get("WorkflowClass", "Test")}Tests.cs',
        'php': f'{info.get("WorkflowClass", "Test")}Test.php',
        'ruby': f'{test_type}_spec.rb'
    }
    
    test_filename = test_filename_patterns.get(language, f'test_{test_type}{ext}')
    test_path = os.path.join(output_dir, test_filename)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write test file
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"Generated: {test_path}")
    return test_path


def main():
    parser = argparse.ArgumentParser(description='Generate unit tests for Temporal implementations')
    parser.add_argument('--implementation', required=True, help='Path to implementation file')
    parser.add_argument('--type', required=True, choices=['workflow', 'activity', 'worker'],
                       help='Type of implementation')
    parser.add_argument('--category', required=True, 
                       help='Category: api/internal, api/external, or service')
    parser.add_argument('--name', required=True, help='API or service name')
    parser.add_argument('--output', required=True, help='Output directory for tests')
    
    args = parser.parse_args()
    
    # Detect language
    language = detect_language(args.implementation)
    if not language:
        print(f"Error: Could not detect language from {args.implementation}")
        sys.exit(1)
    
    print(f"Generating {language} tests for {args.type}: {args.name}")
    print(f"Category: {args.category}")
    
    # Generate tests
    generated_files = []
    
    # Generate workflow/activity tests
    if args.type in ['workflow', 'activity']:
        test_file = generate_test_file(
            args.implementation,
            args.type,
            language,
            args.output,
            args.name
        )
        if test_file:
            generated_files.append(test_file)
    
    # Generate integration test
    integration_file = generate_test_file(
        args.implementation,
        'integration',
        language,
        args.output,
        args.name
    )
    if integration_file:
        generated_files.append(integration_file)
    
    # Summary
    print(f"\n✅ Generated {len(generated_files)} test files:")
    for f in generated_files:
        print(f"  - {f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
