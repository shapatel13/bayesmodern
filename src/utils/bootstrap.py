from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_src_path(src_path: Path) -> None:
    resolved = str(src_path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def prefer_local_package(package_name: str, package_dir: Path) -> None:
    package_dir = package_dir.resolve()
    init_file = package_dir / "__init__.py"
    existing = sys.modules.get(package_name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == init_file:
        return

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local package `{package_name}` from {package_dir}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
