#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Test for NFL DFS Optimizer Import Issues
Run with: python diagnose_import.py
"""

import sys
import traceback

print("="*70)
print("NFL DFS OPTIMIZER - IMPORT DIAGNOSTIC")
print("="*70)
print(f"\nPython Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print()

# Test 1: Basic File Read
print("TEST 1: Can we read the file?")
print("-"*70)
try:
    with open('nfl_dfs_optimizer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"✓ File read successfully ({len(content)} characters)")
except Exception as e:
    print(f"✗ Cannot read file: {e}")
    sys.exit(1)

# Test 2: Syntax Check
print("\nTEST 2: Does the file have valid Python syntax?")
print("-"*70)
try:
    compile(content, 'nfl_dfs_optimizer.py', 'exec')
    print("✓ No syntax errors found")
except SyntaxError as e:
    print(f"✗ SYNTAX ERROR FOUND!")
    print(f"\n  Location: Line {e.lineno}, Column {e.offset}")
    print(f"  Error: {e.msg}")
    if e.text:
        print(f"\n  Problematic line:")
        print(f"  {e.lineno}: {e.text.rstrip()}")
        if e.offset:
            print(f"  {' ' * (len(str(e.lineno)) + 2 + e.offset - 1)}^")
    print(f"\n  Full error:\n")
    traceback.print_exc()
    print("\n" + "="*70)
    print("FIX: Open nfl_dfs_optimizer.py and fix the syntax error above")
    print("="*70)
    sys.exit(1)

# Test 3: Module Import
print("\nTEST 3: Can we import the module?")
print("-"*70)
try:
    import nfl_dfs_optimizer
    print("✓ Module imported successfully")
except SyntaxError as e:
    print(f"✗ Syntax error during import at line {e.lineno}")
    print(f"  {e.msg}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"✗ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check Available Components
print("\nTEST 4: What components are available?")
print("-"*70)

available_components = {
    'Core Classes': [
        'DraftKingsRules',
        'LineupConstraints',
        'ValidationResult',
        'SimulationResults',
        'AIRecommendation',
    ],
    'Utilities': [
        'OptimizerLogger',
        'PerformanceMonitor',
        'UnifiedCache',
        'get_logger',
    ],
    'Data Processing': [
        'OptimizedDataProcessor',
        'PlayerPoolAnalyzer',
        'DataValidator',
    ],
    'Algorithms': [
        'GeneticAlgorithmOptimizer',
        'StandardLineupOptimizer',
        'SimulatedAnnealingOptimizer',
        'SmartGreedyOptimizer',
        'EnsembleOptimizer',
    ],
    'AI Components': [
        'GameTheoryStrategist',
        'CorrelationStrategist',
        'ContrarianStrategist',
        'StackingExpertStrategist',
        'LeverageSpecialist',
    ],
    'Master': [
        'MasterOptimizer',
        'optimize_showdown',
    ],
}

import nfl_dfs_optimizer

results = {}
for category, components in available_components.items():
    results[category] = []
    for component in components:
        if hasattr(nfl_dfs_optimizer, component):
            results[category].append(f"✓ {component}")
        else:
            results[category].append(f"✗ {component} (missing)")

for category, items in results.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")

# Test 5: Check for Common Issues
print("\n" + "="*70)
print("TEST 5: Checking for common issues...")
print("-"*70)

issues_found = []

# Check for smart quotes
if '"' in content or '"' in content or ''' in content or ''' in content:
    issues_found.append("⚠ Found smart quotes - replace with regular quotes")

# Check for problematic Unicode
problematic_chars = []
for i, line in enumerate(content.split('\n'), 1):
    for char in line:
        if ord(char) > 127 and char not in ['\n', '\r']:
            if char not in ['✓', '✗', '⚠', '€', '£', '¥']:  # Common acceptable ones
                problematic_chars.append((i, char, ord(char)))
                if len(problematic_chars) >= 5:  # Limit to first 5
                    break
    if len(problematic_chars) >= 5:
        break

if problematic_chars:
    issues_found.append(f"⚠ Found special Unicode characters:")
    for line_num, char, code in problematic_chars:
        issues_found.append(f"  Line {line_num}: '{char}' (code {code})")

# Check encoding declaration
first_line = content.split('\n')[0]
if 'coding' not in first_line and 'utf' not in first_line:
    issues_found.append("⚠ Missing encoding declaration at top of file")
    issues_found.append("  Add: # -*- coding: utf-8 -*-")

if not issues_found:
    print("✓ No common issues found")
else:
    for issue in issues_found:
        print(issue)

# Final Summary
print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

if not issues_found:
    print("\n✓ All tests passed! The optimizer should work.")
    print("\nIf Streamlit still has issues:")
    print("  1. Restart Streamlit: streamlit run streamlit_app.py")
    print("  2. Clear cache: streamlit cache clear")
    print("  3. Check that both files are in the same directory")
else:
    print("\n⚠ Issues found that may cause import problems.")
    print("Review the warnings above and fix them.")

print("\n" + "="*70)
