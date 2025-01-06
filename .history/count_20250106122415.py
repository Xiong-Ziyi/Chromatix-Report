import pkgutil
import chromatix
import inspect

def find_submodules(package):
    """
    Recursively find all submodules of a given package.
    """
    submodules = []
    if not hasattr(package, '__path__'):
        # It's a single module, not a package
        return submodules
    
    # Walk through the package to find submodules
    for importer, mod_name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + mod_name
        try:
            submodule = importlib.import_module(full_name)
        except ImportError:
            # Skip modules that fail to import
            continue
        
        submodules.append(submodule)
        if is_pkg:
            submodules.extend(find_submodules(submodule))
    
    return submodules

def count_classes_and_functions(module):
    """
    Count classes and functions defined in a single module.
    """
    classes = 0
    functions = 0
    
    # Get all attributes in the module
    for name, obj in inspect.getmembers(module):
        # Ensure the object's __module__ matches the module's name 
        # (to avoid counting imported classes/functions from other modules).
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            classes += 1
        elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
            functions += 1

    return classes, functions

def count_in_package(package):
    """
    Count total classes and functions across a package and all submodules.
    """
    # Start with the root module itself
    modules_to_check = [package]
    
    # Recursively find all submodules
    modules_to_check.extend(find_submodules(package))
    
    total_classes = 0
    total_functions = 0
    
    # Count for each discovered submodule
    for mod in modules_to_check:
        c, f = count_classes_and_functions(mod)
        total_classes += c
        total_functions += f
    
    return total_classes, total_functions


# ---- Example Usage ----
if __name__ == "__main__":
    # Suppose you want to analyze the 'requests' library
    import requests
    
    cls_count, func_count = count_in_package(requests)
    print("Classes:", cls_count)
    print("Functions:", func_count)
