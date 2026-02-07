import importlib

def test_basic_imports():
    modules = [
        "datasets",
        "models",
        "models.backbones",
        "models.heads",
        "augments",
        "engine",
    ]
    for name in modules:
        importlib.import_module(name)
