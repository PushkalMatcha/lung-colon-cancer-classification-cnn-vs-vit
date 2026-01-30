"""Utilities for reading and writing the combined results JSON.

This module centralizes access to `results/all_models_history.json` and provides
safe atomic writes plus a tiny CLI to list and show model entries.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Optional


DEFAULT_RESULTS_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'all_models_history.json'))


def load_all_results(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the combined results JSON. Returns an empty dict if the file doesn't exist or is invalid."""
    path = path or DEFAULT_RESULTS_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_all_results(data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Atomically write the full results dict to disk with pretty formatting."""
    path = path or DEFAULT_RESULTS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # atomic write to temporary file then move
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix='._tmp_results_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        shutil.move(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def save_model_result(model_name: str, overall_accuracy: float, classification_report: Dict[str, Any], confusion_matrix: Any, path: Optional[str] = None, overwrite: bool = True) -> None:
    """Add or update a single model entry in the combined results JSON.

    confusion_matrix may be a numpy array or nested lists; this will try to convert
    it to native Python lists.
    """
    all_results = load_all_results(path)

    # Normalize confusion matrix to lists
    try:
        # avoid importing numpy at top-level for portability
        import numpy as _np

        if isinstance(confusion_matrix, _np.ndarray):
            cm_list = confusion_matrix.tolist()
        else:
            cm_list = confusion_matrix
    except Exception:
        cm_list = confusion_matrix

    entry = {
        "overall_accuracy": float(round(overall_accuracy, 4)),
        "classification_report": classification_report,
        "confusion_matrix": cm_list,
    }

    if not overwrite and model_name in all_results:
        raise KeyError(f"Model '{model_name}' already exists in results and overwrite=False")

    all_results[model_name] = entry
    save_all_results(all_results, path)


def list_models(path: Optional[str] = None) -> Dict[str, Any]:
    """Return the all-results dict (keys are model names)."""
    return load_all_results(path)


def get_model_entry(model_name: str, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    all_results = load_all_results(path)
    return all_results.get(model_name)


def print_summary(path: Optional[str] = None) -> None:
    all_results = load_all_results(path)
    if not all_results:
        print(f"No results found at '{path or DEFAULT_RESULTS_PATH}'")
        return
    for name, data in all_results.items():
        acc = data.get('overall_accuracy')
        print(f"- {name}: overall_accuracy={acc}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description='Inspect or update combined model results JSON')
    parser.add_argument('--list', action='store_true', help='List models stored in the results JSON')
    parser.add_argument('--show', type=str, metavar='MODEL', help='Print the JSON entry for MODEL')
    parser.add_argument('--path', type=str, default=None, help='Optional path to results JSON')
    args = parser.parse_args()

    if args.list:
        all_results = list_models(args.path)
        if not all_results:
            print('No results found')
            return
        for name in sorted(all_results.keys()):
            print(name)
        return

    if args.show:
        entry = get_model_entry(args.show, args.path)
        if entry is None:
            print(f"No entry for model '{args.show}'")
            return
        print(json.dumps(entry, indent=4, ensure_ascii=False))
        return

    # default action: print brief summary
    print_summary(args.path)


if __name__ == '__main__':
    _cli()
