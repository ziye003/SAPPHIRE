#!/usr/bin/env python3
"""
run_all.py
==========
Master script: runs the main SAPPHIRE validation pipeline and/or the
robustness-check suite, in sequence.

This repo's scripts are written as exec()-style notebook cells (see each
script's own docstring), not as independently importable modules, so this
runner loads each one with exec() into a dedicated namespace and calls its
public entry point directly. A dedicated namespace (rather than this
module's own globals()) is used so that loading e.g. holdout_validation.py
cannot silently overwrite this file's own main(), and __name__ is set to a
sentinel value so that each script's own `if __name__ == "__main__":` block
does not also fire during the exec().

Usage:
------
python run_all.py                       # run everything
python run_all.py --sapphire-only       # run only the main validation pipeline
python run_all.py --robustness-only     # run only the robustness-check suite
python run_all.py --single-dataset Cardiomyocyte   # main pipeline, one dataset only
                                                     # (robustness scripts always run
                                                     # across all datasets)
"""

import argparse
import os

_here = os.path.dirname(os.path.abspath(__file__))

# Shared namespace for all exec()'d scripts -- keeps sapphire_core's symbols
# (DATA_ROOT, DATASETS_CONFIG, load_and_prepare, ...) available across
# scripts without polluting this file's own globals().
_ns = {"__name__": "sapphire_run_all"}


def _load(script_name):
    """exec() a script from this directory into the shared namespace _ns."""
    path = os.path.join(_here, script_name)
    print(f"  -> Loading {script_name}...")
    exec(open(path).read(), _ns)


def run_main_pipeline(single_dataset=None):
    print("\n" + "=" * 80)
    print("RUNNING SAPPHIRE MAIN VALIDATION PIPELINE")
    print("=" * 80 + "\n")
    if "load_and_prepare" not in _ns:
        _load("sapphire_core.py")
    _load("sapphire_validation_all.py")
    _ns["run_pipeline"](dataset=single_dataset)


def run_robustness_suite():
    print("\n" + "=" * 80)
    print("RUNNING ROBUSTNESS CHECKS")
    print("=" * 80 + "\n")
    if "load_and_prepare" not in _ns:
        _load("sapphire_core.py")

    print("\n--- Holdout-cell validation ---")
    _load("holdout_validation.py")
    _ns["main"]()

    print("\n--- Hyperparameter sensitivity ---")
    _load("hyperparameter_sensitivity.py")
    _ns["run_hyperparameter_sensitivity"]()

    print("\n--- Resampling stability ---")
    _load("resampling_stability.py")
    _ns["run_resampling_stability"]()

    print("\n--- Read-depth control ---")
    _load("read_depth_control.py")
    _ns["run_read_depth_control"]()


def main():
    parser = argparse.ArgumentParser(description="Run SAPPHIRE pipelines")
    parser.add_argument("--sapphire-only", action="store_true",
                       help="Run only the main validation pipeline")
    parser.add_argument("--robustness-only", action="store_true",
                       help="Run only the robustness-check suite")
    parser.add_argument("--single-dataset", type=str, default=None,
                       help='Restrict the main pipeline to one dataset '
                            '(e.g. "Cardiomyocyte"). Robustness scripts '
                            "always run across all datasets.")
    args = parser.parse_args()

    if not args.robustness_only:
        run_main_pipeline(single_dataset=args.single_dataset)

    if not args.sapphire_only:
        run_robustness_suite()

    print("\n" + "=" * 80)
    print("ALL REQUESTED PIPELINES COMPLETE")
    print("=" * 80)
    print(f"\nCheck outputs under: {_ns['DATA_ROOT']}")


if __name__ == "__main__":
    main()
