#!/usr/bin/env python3
"""
Temporal Integration System - Test Runner
Runs generated tests and captures results

Usage:
    python run-tests.py --test-dir <directory> --language <lang> --output <results.json>
"""

import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional


# Test framework commands for each language
TEST_COMMANDS = {
    'python': {
        'command': ['pytest', '--verbose', '--tb=short', '--cov=.', '--cov-report=json'],
        'coverage_file': 'coverage.json'
    },
    'typescript': {
        'command': ['npm', 'test', '--', '--coverage', '--json', '--outputFile=test-results.json'],
        'coverage_file': 'coverage/coverage-summary.json'
    },
    'go': {
        'command': ['go', 'test', '-v', '-cover', '-json'],
        'coverage_file': None
    },
    'java': {
        'command': ['mvn', 'test'],
        'coverage_file': 'target/site/jacoco/jacoco.xml'
    },
    'csharp': {
        'command': ['dotnet', 'test', '--logger', 'trx', '--collect', 'Code Coverage'],
        'coverage_file': None
    },
    'php': {
        'command': ['phpunit', '--testdox', '--coverage-text'],
        'coverage_file': 'coverage/coverage.xml'
    },
    'ruby': {
        'command': ['rspec', '--format', 'json', '--out', 'rspec-results.json'],
        'coverage_file': 'coverage/.resultset.json'
    }
}


def detect_test_framework(test_dir: str, language: str) -> Optional[str]:
    """Detect which test framework is being used"""
    
    # Check for framework-specific files
    framework_indicators = {
        'python': {
            'pytest': ['pytest.ini', 'pyproject.toml', 'setup.cfg'],
            'unittest': []
        },
        'typescript': {
            'jest': ['jest.config.js', 'jest.config.ts'],
            'vitest': ['vitest.config.ts', 'vitest.config.js'],
            'mocha': ['.mocharc.json', 'mocha.opts']
        }
    }
    
    if language in framework_indicators:
        for framework, indicator_files in framework_indicators[language].items():
            for indicator in indicator_files:
                if os.path.exists(os.path.join(test_dir, '..', '..', indicator)):
                    return framework
    
    # Default frameworks
    defaults = {
        'python': 'pytest',
        'typescript': 'jest',
        'go': 'go test',
        'java': 'junit',
        'csharp': 'xunit',
        'php': 'phpunit',
        'ruby': 'rspec'
    }
    
    return defaults.get(language)


def run_python_tests(test_dir: str) -> Dict:
    """Run Python tests with pytest"""
    results = {
        'language': 'python',
        'test_dir': test_dir,
        'framework': 'pytest',
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'duration': 0,
        'coverage': 0,
        'test_files': []
    }
    
    start_time = time.time()
    
    try:
        # Run pytest
        cmd = ['pytest', test_dir, '--verbose', '--tb=short', '--json-report', 
               '--json-report-file=test-report.json', '--cov=temporal', 
               '--cov-report=json:coverage.json', '--cov-report=term']
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(test_dir) or '.'
        )
        
        results['duration'] = time.time() - start_time
        results['exit_code'] = result.returncode
        results['stdout'] = result.stdout
        results['stderr'] = result.stderr
        
        # Parse pytest JSON report if available
        report_file = 'test-report.json'
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report = json.load(f)
                
                if 'summary' in report:
                    results['total_tests'] = report['summary'].get('total', 0)
                    results['passed'] = report['summary'].get('passed', 0)
                    results['failed'] = report['summary'].get('failed', 0)
                    results['skipped'] = report['summary'].get('skipped', 0)
                
                if 'tests' in report:
                    # Group by file
                    files = {}
                    for test in report['tests']:
                        file_path = test.get('nodeid', '').split('::')[0]
                        if file_path not in files:
                            files[file_path] = {
                                'file': file_path,
                                'tests': 0,
                                'passed': 0,
                                'failed': 0,
                                'duration': 0
                            }
                        
                        files[file_path]['tests'] += 1
                        files[file_path]['duration'] += test.get('duration', 0)
                        
                        if test.get('outcome') == 'passed':
                            files[file_path]['passed'] += 1
                        elif test.get('outcome') == 'failed':
                            files[file_path]['failed'] += 1
                    
                    results['test_files'] = list(files.values())
        
        # Parse coverage
        coverage_file = 'coverage.json'
        if os.path.exists(coverage_file):
            with open(coverage_file, 'r') as f:
                coverage = json.load(f)
                if 'totals' in coverage:
                    results['coverage'] = coverage['totals'].get('percent_covered', 0)
        
        # Fallback: parse from stdout
        if results['total_tests'] == 0:
            # Look for pytest summary line
            for line in result.stdout.split('\n'):
                if 'passed' in line or 'failed' in line:
                    import re
                    passed_match = re.search(r'(\d+) passed', line)
                    failed_match = re.search(r'(\d+) failed', line)
                    skipped_match = re.search(r'(\d+) skipped', line)
                    
                    if passed_match:
                        results['passed'] = int(passed_match.group(1))
                    if failed_match:
                        results['failed'] = int(failed_match.group(1))
                    if skipped_match:
                        results['skipped'] = int(skipped_match.group(1))
                    
                    results['total_tests'] = results['passed'] + results['failed'] + results['skipped']
                    break
        
        # Parse coverage from stdout if not found
        if results['coverage'] == 0:
            for line in result.stdout.split('\n'):
                if 'TOTAL' in line and '%' in line:
                    import re
                    match = re.search(r'(\d+)%', line)
                    if match:
                        results['coverage'] = float(match.group(1))
                        break
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def run_typescript_tests(test_dir: str) -> Dict:
    """Run TypeScript tests with Jest"""
    results = {
        'language': 'typescript',
        'test_dir': test_dir,
        'framework': 'jest',
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'duration': 0,
        'coverage': 0,
        'test_files': []
    }
    
    start_time = time.time()
    
    try:
        # Run jest
        cmd = ['npx', 'jest', test_dir, '--json', '--coverage', '--testLocationInResults']
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(test_dir) or '.'
        )
        
        results['duration'] = time.time() - start_time
        results['exit_code'] = result.returncode
        
        # Parse Jest JSON output
        try:
            jest_results = json.loads(result.stdout)
            
            results['total_tests'] = jest_results.get('numTotalTests', 0)
            results['passed'] = jest_results.get('numPassedTests', 0)
            results['failed'] = jest_results.get('numFailedTests', 0)
            results['skipped'] = jest_results.get('numPendingTests', 0)
            
            # Parse test files
            if 'testResults' in jest_results:
                for test_file in jest_results['testResults']:
                    file_result = {
                        'file': test_file.get('name', ''),
                        'tests': len(test_file.get('assertionResults', [])),
                        'passed': sum(1 for t in test_file.get('assertionResults', []) 
                                    if t.get('status') == 'passed'),
                        'failed': sum(1 for t in test_file.get('assertionResults', []) 
                                    if t.get('status') == 'failed'),
                        'duration': test_file.get('perfStats', {}).get('runtime', 0) / 1000
                    }
                    results['test_files'].append(file_result)
            
            # Parse coverage
            if 'coverageMap' in jest_results:
                # Calculate average coverage
                coverages = []
                for file_coverage in jest_results['coverageMap'].values():
                    if 'lines' in file_coverage:
                        pct = file_coverage['lines'].get('pct', 0)
                        coverages.append(pct)
                
                if coverages:
                    results['coverage'] = sum(coverages) / len(coverages)
        
        except json.JSONDecodeError:
            results['error'] = 'Could not parse Jest output'
            results['stdout'] = result.stdout
            results['stderr'] = result.stderr
    
    except Exception as e:
        results['error'] = str(e)
    
    return results


def run_go_tests(test_dir: str) -> Dict:
    """Run Go tests"""
    results = {
        'language': 'go',
        'test_dir': test_dir,
        'framework': 'testing',
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'duration': 0,
        'coverage': 0,
        'test_files': []
    }
    
    start_time = time.time()
    
    try:
        # Run go test
        cmd = ['go', 'test', '-v', '-cover', '-json', './...']
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=test_dir
        )
        
        results['duration'] = time.time() - start_time
        results['exit_code'] = result.returncode
        
        # Parse JSON output
        for line in result.stdout.split('\n'):
            if not line.strip():
                continue
            
            try:
                event = json.loads(line)
                
                if event.get('Action') == 'pass':
                    results['passed'] += 1
                elif event.get('Action') == 'fail':
                    results['failed'] += 1
                elif event.get('Action') == 'skip':
                    results['skipped'] += 1
            except:
                pass
        
        results['total_tests'] = results['passed'] + results['failed'] + results['skipped']
        
        # Parse coverage from output
        for line in result.stdout.split('\n'):
            if 'coverage:' in line:
                import re
                match = re.search(r'coverage:\s+([\d.]+)%', line)
                if match:
                    results['coverage'] = float(match.group(1))
                    break
    
    except Exception as e:
        results['error'] = str(e)
    
    return results


def run_java_tests(test_dir: str) -> Dict:
    """Run Java tests with Maven or Gradle"""
    results = {
        'language': 'java',
        'test_dir': test_dir,
        'framework': 'junit',
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'duration': 0,
        'coverage': 0,
        'test_files': []
    }
    
    start_time = time.time()
    
    try:
        # Detect Maven or Gradle
        if os.path.exists('pom.xml'):
            cmd = ['mvn', 'test']
        elif os.path.exists('build.gradle'):
            cmd = ['gradle', 'test']
        else:
            results['error'] = 'No Maven or Gradle project found'
            return results
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=test_dir
        )
        
        results['duration'] = time.time() - start_time
        results['exit_code'] = result.returncode
        
        # Parse test results (basic parsing)
        import re
        for line in result.stdout.split('\n'):
            test_match = re.search(r'Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)', line)
            if test_match:
                total = int(test_match.group(1))
                failures = int(test_match.group(2))
                errors = int(test_match.group(3))
                skipped = int(test_match.group(4))
                
                results['total_tests'] = total
                results['failed'] = failures + errors
                results['skipped'] = skipped
                results['passed'] = total - failures - errors - skipped
                break
    
    except Exception as e:
        results['error'] = str(e)
    
    return results


def run_csharp_tests(test_dir: str) -> Dict:
    """Run .NET tests"""
    results = {
        'language': 'csharp',
        'test_dir': test_dir,
        'framework': 'xunit',
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'duration': 0,
        'coverage': 0,
        'test_files': []
    }
    
    start_time = time.time()
    
    try:
        cmd = ['dotnet', 'test', '--logger', 'trx']
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=test_dir
        )
        
        results['duration'] = time.time() - start_time
        results['exit_code'] = result.returncode
        
        # Parse test results
        import re
        for line in result.stdout.split('\n'):
            passed_match = re.search(r'Passed!\s*-\s*Failed:\s*(\d+),\s*Passed:\s*(\d+),\s*Skipped:\s*(\d+),\s*Total:\s*(\d+)', line)
            if passed_match:
                results['failed'] = int(passed_match.group(1))
                results['passed'] = int(passed_match.group(2))
                results['skipped'] = int(passed_match.group(3))
                results['total_tests'] = int(passed_match.group(4))
                break
    
    except Exception as e:
        results['error'] = str(e)
    
    return results


def run_php_tests(test_dir: str) -> Dict:
    """Run PHP tests with PHPUnit"""
    results = {
        'language': 'php',
        'test_dir': test_dir,
        'framework': 'phpunit',
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'duration': 0,
        'coverage': 0,
        'test_files': []
    }
    
    start_time = time.time()
    
    try:
        cmd = ['phpunit', '--testdox', test_dir]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        results['duration'] = time.time() - start_time
        results['exit_code'] = result.returncode
        
        # Parse results
        import re
        ok_match = re.search(r'OK\s*\((\d+) tests?, (\d+) assertions?\)', result.stdout)
        if ok_match:
            results['total_tests'] = int(ok_match.group(1))
            results['passed'] = int(ok_match.group(1))
        
        failures_match = re.search(r'FAILURES!\s*Tests:\s*(\d+),\s*Assertions:\s*\d+,\s*Failures:\s*(\d+)', result.stdout)
        if failures_match:
            results['total_tests'] = int(failures_match.group(1))
            results['failed'] = int(failures_match.group(2))
            results['passed'] = results['total_tests'] - results['failed']
    
    except Exception as e:
        results['error'] = str(e)
    
    return results


def run_ruby_tests(test_dir: str) -> Dict:
    """Run Ruby tests with RSpec"""
    results = {
        'language': 'ruby',
        'test_dir': test_dir,
        'framework': 'rspec',
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'duration': 0,
        'coverage': 0,
        'test_files': []
    }
    
    start_time = time.time()
    
    try:
        cmd = ['rspec', test_dir, '--format', 'json']
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        results['duration'] = time.time() - start_time
        results['exit_code'] = result.returncode
        
        # Parse JSON output
        try:
            rspec_results = json.loads(result.stdout)
            
            summary = rspec_results.get('summary', {})
            results['total_tests'] = summary.get('example_count', 0)
            results['failed'] = summary.get('failure_count', 0)
            results['skipped'] = summary.get('pending_count', 0)
            results['passed'] = results['total_tests'] - results['failed'] - results['skipped']
            results['duration'] = summary.get('duration', 0)
        
        except json.JSONDecodeError:
            results['error'] = 'Could not parse RSpec output'
    
    except Exception as e:
        results['error'] = str(e)
    
    return results


def run_tests(test_dir: str, language: str) -> Dict:
    """Run tests for the specified language"""
    
    runners = {
        'python': run_python_tests,
        'typescript': run_typescript_tests,
        'javascript': run_typescript_tests,
        'go': run_go_tests,
        'java': run_java_tests,
        'csharp': run_csharp_tests,
        'php': run_php_tests,
        'ruby': run_ruby_tests
    }
    
    runner = runners.get(language)
    if not runner:
        return {
            'error': f'Unsupported language: {language}',
            'language': language,
            'test_dir': test_dir
        }
    
    return runner(test_dir)


def main():
    parser = argparse.ArgumentParser(description='Run Temporal implementation tests')
    parser.add_argument('--test-dir', required=True, help='Directory containing tests')
    parser.add_argument('--language', required=True, 
                       choices=['python', 'typescript', 'javascript', 'go', 'java', 'csharp', 'php', 'ruby'],
                       help='Programming language')
    parser.add_argument('--output', required=True, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print(f"Running {args.language} tests in: {args.test_dir}")
    
    # Run tests
    results = run_tests(args.test_dir, args.language)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Results Summary")
    print(f"{'='*60}")
    print(f"Total Tests: {results.get('total_tests', 0)}")
    print(f"Passed: {results.get('passed', 0)}")
    print(f"Failed: {results.get('failed', 0)}")
    print(f"Skipped: {results.get('skipped', 0)}")
    print(f"Duration: {results.get('duration', 0):.2f}s")
    print(f"Coverage: {results.get('coverage', 0):.1f}%")
    
    if 'error' in results:
        print(f"\n⚠️  Error: {results['error']}")
        return 1
    
    if results.get('failed', 0) > 0:
        print(f"\n❌ Tests failed: {results['failed']}/{results['total_tests']}")
        return 1
    
    print(f"\n✅ All tests passed!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
