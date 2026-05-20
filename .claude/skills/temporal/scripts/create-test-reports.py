#!/usr/bin/env python3
"""
Temporal Integration System - Test Report Generator
Generates markdown test reports from test results JSON

Usage:
    python create-test-reports.py --results <test-results.json> --output <REPORT.md>
    python create-test-reports.py --summary <test_dir> --output <SUMMARY.md>
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def get_status_emoji(passed: int, failed: int, total: int) -> str:
    """Get status emoji based on test results"""
    if failed == 0 and total > 0:
        return "✅"
    elif failed > 0:
        return "❌"
    else:
        return "⚠️"


def generate_recommendations(results: Dict) -> List[str]:
    """Generate recommendations based on test results"""
    recommendations = []
    
    # Coverage recommendations
    coverage = results.get('coverage', 0)
    if coverage < 60:
        recommendations.append("⚠️ Coverage is below 60%. Consider adding more tests.")
    elif coverage < 80:
        recommendations.append("ℹ️ Coverage is good but could be improved (target: 80%+)")
    else:
        recommendations.append("✅ Excellent test coverage!")
    
    # Failed tests
    failed = results.get('failed', 0)
    if failed > 0:
        recommendations.append(f"❌ {failed} test(s) failed. Review and fix failures.")
    else:
        recommendations.append("✅ All tests passing!")
    
    # Test count
    total = results.get('total_tests', 0)
    if total < 5:
        recommendations.append("ℹ️ Consider adding more test cases for better coverage")
    
    # Performance
    duration = results.get('duration', 0)
    if duration > 30:
        recommendations.append("⚠️ Tests are taking a long time. Consider optimizing.")
    
    # Missing tests
    test_files = results.get('test_files', [])
    if len(test_files) < 3:
        recommendations.append("ℹ️ Consider adding integration tests")
    
    return recommendations


def generate_test_report(results: Dict, output_file: str, api_name: str) -> None:
    """Generate markdown test report for a single implementation"""
    
    # Extract data
    status_emoji = get_status_emoji(
        results.get('passed', 0),
        results.get('failed', 0),
        results.get('total_tests', 0)
    )
    
    status_text = "PASSED" if results.get('failed', 0) == 0 else "FAILED"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build report
    report = f"""# Test Report: {api_name}

**Generated:** {timestamp}  
**Status:** {status_emoji} {status_text} ({results.get('passed', 0)}/{results.get('total_tests', 0)} tests)  
**Duration:** {format_duration(results.get('duration', 0))}  
**Coverage:** {results.get('coverage', 0):.1f}%  
**Language:** {results.get('language', 'unknown')}  
**Framework:** {results.get('framework', 'unknown')}

---

## Test Summary

| File | Tests | Passed | Failed | Duration |
|------|-------|--------|--------|----------|
"""
    
    # Add test files
    test_files = results.get('test_files', [])
    if test_files:
        for test_file in test_files:
            file_name = Path(test_file.get('file', '')).name or 'unknown'
            tests = test_file.get('tests', 0)
            passed = test_file.get('passed', 0)
            failed = test_file.get('failed', 0)
            duration = format_duration(test_file.get('duration', 0))
            
            status = "✅" if failed == 0 else "❌"
            report += f"| {status} {file_name} | {tests} | {passed} | {failed} | {duration} |\n"
    else:
        report += f"| All tests | {results.get('total_tests', 0)} | {results.get('passed', 0)} | {results.get('failed', 0)} | {format_duration(results.get('duration', 0))} |\n"
    
    report += "\n---\n\n"
    
    # Add detailed results section
    report += "## Detailed Results\n\n"
    
    if results.get('failed', 0) > 0:
        report += "### ❌ Failed Tests\n\n"
        if 'stderr' in results and results['stderr']:
            report += "```\n"
            report += results['stderr'][:1000]  # Limit stderr output
            if len(results['stderr']) > 1000:
                report += "\n... (truncated)\n"
            report += "```\n\n"
        else:
            report += "See test output for details.\n\n"
    
    if results.get('skipped', 0) > 0:
        report += f"### ⚠️ Skipped Tests\n\n"
        report += f"{results['skipped']} test(s) were skipped.\n\n"
    
    # Coverage details
    report += "## Coverage Analysis\n\n"
    coverage = results.get('coverage', 0)
    
    if coverage >= 80:
        report += f"✅ **Excellent coverage:** {coverage:.1f}%\n\n"
    elif coverage >= 60:
        report += f"⚠️ **Good coverage:** {coverage:.1f}% (target: 80%+)\n\n"
    else:
        report += f"❌ **Low coverage:** {coverage:.1f}% (target: 80%+)\n\n"
    
    report += "Consider adding tests for:\n"
    report += "- Edge cases and error handling\n"
    report += "- Timeout and retry scenarios\n"
    report += "- Integration tests with real dependencies\n\n"
    
    # Recommendations
    recommendations = generate_recommendations(results)
    if recommendations:
        report += "---\n\n"
        report += "## Recommendations\n\n"
        for rec in recommendations:
            report += f"- {rec}\n"
        report += "\n"
    
    # Error details
    if 'error' in results:
        report += "---\n\n"
        report += "## ⚠️ Errors\n\n"
        report += f"```\n{results['error']}\n```\n\n"
    
    # Metadata
    report += "---\n\n"
    report += "## Test Metadata\n\n"
    report += f"- **Test Directory:** `{results.get('test_dir', 'unknown')}`\n"
    report += f"- **Exit Code:** {results.get('exit_code', 'unknown')}\n"
    report += f"- **Report Generated:** {timestamp}\n"
    
    # Write report
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Generated report: {output_file}")


def generate_summary_report(test_dir: str, output_file: str) -> None:
    """Generate overall summary report for all tests"""
    
    # Find all REPORT.md files in test directory
    test_reports = []
    
    for root, dirs, files in os.walk(test_dir):
        if 'REPORT.md' in files:
            report_path = os.path.join(root, 'REPORT.md')
            
            # Try to find corresponding test-results.json
            results_path = os.path.join(root, 'test-results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    
                    # Get relative path for category
                    rel_path = os.path.relpath(root, test_dir)
                    
                    test_reports.append({
                        'path': rel_path,
                        'results': results,
                        'report_path': report_path
                    })
    
    # Generate summary
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""# Temporal Tests - Overall Summary

**Generated:** {timestamp}  
**Total Implementations Tested:** {len(test_reports)}

---

"""
    
    # Calculate overall status
    total_implementations = len(test_reports)
    passed_implementations = sum(1 for r in test_reports if r['results'].get('failed', 0) == 0)
    failed_implementations = total_implementations - passed_implementations
    
    if failed_implementations == 0 and total_implementations > 0:
        overall_status = "✅ ALL PASSED"
    elif failed_implementations > 0:
        overall_status = f"❌ {failed_implementations} FAILED"
    else:
        overall_status = "⚠️ NO TESTS"
    
    summary += f"**Overall Status:** {overall_status}\n\n"
    
    # Summary statistics
    total_tests = sum(r['results'].get('total_tests', 0) for r in test_reports)
    total_passed = sum(r['results'].get('passed', 0) for r in test_reports)
    total_failed = sum(r['results'].get('failed', 0) for r in test_reports)
    total_duration = sum(r['results'].get('duration', 0) for r in test_reports)
    avg_coverage = sum(r['results'].get('coverage', 0) for r in test_reports) / len(test_reports) if test_reports else 0
    
    summary += f"""## Overall Statistics

- **Total Tests:** {total_tests}
- **Passed:** {total_passed}
- **Failed:** {total_failed}
- **Total Duration:** {format_duration(total_duration)}
- **Average Coverage:** {avg_coverage:.1f}%

---

## Summary by Implementation

| Implementation | Status | Tests | Duration | Coverage | Report |
|---------------|--------|-------|----------|----------|--------|
"""
    
    # Sort by category and name
    test_reports.sort(key=lambda x: x['path'])
    
    for report in test_reports:
        path = report['path']
        results = report['results']
        
        status_emoji = get_status_emoji(
            results.get('passed', 0),
            results.get('failed', 0),
            results.get('total_tests', 0)
        )
        
        tests_str = f"{results.get('passed', 0)}/{results.get('total_tests', 0)}"
        duration = format_duration(results.get('duration', 0))
        coverage = f"{results.get('coverage', 0):.0f}%"
        report_link = f"[View](./{path}/REPORT.md)"
        
        summary += f"| {path} | {status_emoji} | {tests_str} | {duration} | {coverage} | {report_link} |\n"
    
    summary += "\n---\n\n"
    
    # Failed implementations
    failed_reports = [r for r in test_reports if r['results'].get('failed', 0) > 0]
    if failed_reports:
        summary += "## ❌ Failed Implementations\n\n"
        for report in failed_reports:
            summary += f"- **{report['path']}**: {report['results'].get('failed', 0)} test(s) failed\n"
        summary += "\n---\n\n"
    
    # Recommendations
    summary += "## Recommendations\n\n"
    
    if failed_implementations > 0:
        summary += f"- ❌ Fix {failed_implementations} failed implementation(s)\n"
    else:
        summary += "- ✅ All implementations passing!\n"
    
    if avg_coverage < 80:
        summary += f"- ⚠️ Improve overall coverage (current: {avg_coverage:.1f}%, target: 80%+)\n"
    else:
        summary += f"- ✅ Excellent overall coverage: {avg_coverage:.1f}%\n"
    
    if total_duration > 60:
        summary += f"- ⚠️ Total test duration is high ({format_duration(total_duration)}). Consider optimizing.\n"
    
    summary += "\n---\n\n"
    summary += f"*Report generated on {timestamp}*\n"
    
    # Write summary
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Generated summary: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate Temporal test reports')
    parser.add_argument('--results', help='Path to test results JSON file')
    parser.add_argument('--output', required=True, help='Output markdown file')
    parser.add_argument('--name', help='API or service name (for individual reports)')
    parser.add_argument('--summary', help='Test directory (for generating SUMMARY.md)')
    
    args = parser.parse_args()
    
    if args.results:
        # Generate individual report
        if not args.name:
            # Extract name from path
            args.name = Path(args.results).parent.name
        
        print(f"Generating test report for: {args.name}")
        
        with open(args.results, 'r') as f:
            results = json.load(f)
        
        generate_test_report(results, args.output, args.name)
        
    elif args.summary:
        # Generate summary report
        print(f"Generating summary report for: {args.summary}")
        generate_summary_report(args.summary, args.output)
        
    else:
        print("Error: Must specify either --results or --summary")
        return 1
    
    print("✅ Report generation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
