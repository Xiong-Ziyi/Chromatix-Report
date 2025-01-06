import ast
import os

def count_classes_and_functions_in_file(filepath):
    """
    Count number of class and function definitions in a single Python file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    classes = 0
    functions = 0
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes += 1
        elif isinstance(node, ast.FunctionDef):
            functions += 1
            
    return classes, functions

def count_in_source_dir(directory):
    """
    Recursively count classes and functions in all .py files under 'directory'.
    """
    total_classes = 0
    total_functions = 0
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                cls_count, func_count = count_classes_and_functions_in_file(filepath)
                total_classes += cls_count
                total_functions += func_count
    
    return total_classes, total_functions

# ---- Example usage ----
if __name__ == "__main__":
    # Suppose 'my_package' is a directory on disk containing __init__.py etc.
    directory_path = "/path/to/my_package"
    cls_count, func_count = count_in_source_dir(directory_path)
    print("Classes:", cls_count)
    print("Functions:", func_count)
