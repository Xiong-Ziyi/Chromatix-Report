import pkgutil
import importlib
import inspect

def find_all_submodules(package_name):
    """
    Return a list of module objects (including sub-packages)
    under the given top-level package name, e.g. "Chromatix".
    """
    modules = []
    visited = set()

    def walk(name):
        if name in visited:
            return
        visited.add(name)
        
        # Import the module or package
        pkg = importlib.import_module(name)
        modules.append(pkg)

        # If it's a package, iterate its contents
        if hasattr(pkg, '__path__'):  # indicates a package
            for info in pkgutil.iter_modules(pkg.__path__, prefix=name+'.'):
                walk(info.name)

    # Start recursion from the root package
    walk(package_name)
    return modules

def count_classes_and_functions_per_module(modules):
    """
    Given a list of imported module objects, return a dict with:
        {
           "some_pkg": {
               "is_package": True,
               "classes": X,
               "functions": Y
           },
           "some_pkg.submodule": {
               "is_package": False,
               "classes": ...,
               "functions": ...
           },
           ...
        }
    """
    results = {}

    for mod in modules:
        classes = 0
        functions = 0

        # Determine if 'mod' is a package
        is_package = hasattr(mod, '__path__')

        # Inspect all attributes in the module
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and obj.__module__ == mod.__name__:
                classes += 1
            elif inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                functions += 1

        results[mod.__name__] = {
            "is_package": is_package,
            "classes": classes,
            "functions": functions
        }

    return results

if __name__ == "__main__":
    # Example: Analyze the 'Chromatix' library (replace with the real import name).
    package_name = "chromatix"

    # 1. Discover all modules & packages
    modules = find_all_submodules(package_name)

    # 2. Count classes & functions per module
    module_counts = count_classes_and_functions_per_module(modules)

    # 3. Print or process the results
    #    For instance, show each module, how many classes, how many functions,
    #    and whether it is a package.
    for mod_name, info in sorted(module_counts.items()):
        print(f"Module: {mod_name}")
        print(f"  Is Package: {info['is_package']}")
        print(f"  Classes:   {info['classes']}")
        print(f"  Functions: {info['functions']}")
        print()
