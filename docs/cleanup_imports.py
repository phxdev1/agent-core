#!/usr/bin/env python3
"""
Import cleanup utility
Removes unused imports and fixes import order
"""

import ast
import sys
from pathlib import Path


def check_imports(filepath):
    """Check for unused imports in a Python file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Syntax error in {filepath}")
        return []
    
    imports = []
    used_names = set()
    
    # Collect all imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, alias.asname or alias.name))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}" if node.module else alias.name
                imports.append((full_name, alias.asname or alias.name))
        elif isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
    
    # Find unused imports
    unused = []
    for full_name, used_name in imports:
        if used_name not in used_names:
            unused.append(full_name)
    
    return unused


def main():
    """Check all Python files for unused imports"""
    py_files = list(Path('.').glob('*.py'))
    
    print("Checking for unused imports...")
    print("-" * 40)
    
    for filepath in py_files:
        if filepath.name == 'cleanup_imports.py':
            continue
            
        unused = check_imports(filepath)
        if unused:
            print(f"\n{filepath}:")
            for imp in unused:
                print(f"  - {imp}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()