import pkgutil
import importlib

llm_interfaces = []

for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f".{module_name}", package=__name__)

    if hasattr(module, "LLMClass"):
        llm_interfaces.append(module.LLMClass)