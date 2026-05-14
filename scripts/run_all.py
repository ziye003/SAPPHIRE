#!/usr/bin/env python3
"""
MASTER RUN SCRIPT
=================

Run both SAPPHIRE main pipeline and robustness checks

Usage:
------
python run_all.py                    # Run everything
python run_all.py --sapphire-only    # Run only main pipeline
python run_all.py --robustness-only  # Run only robustness checks
"""

import argparse
from pathlib import Path

# Update these paths to match your system
DATA_ROOT = Path("/Users/ziye/Documents/paper/data")

def main():
    parser = argparse.ArgumentParser(description='Run SAPPHIRE pipelines')
    parser.add_argument('--sapphire-only', action='store_true', 
                       help='Run only SAPPHIRE main pipeline')
    parser.add_argument('--robustness-only', action='store_true',
                       help='Run only robustness checks')
    parser.add_argument('--single-dataset', type=str, default=None,
                       help='Run on single dataset (e.g., "Cardiomyocyte")')
    
    args = parser.parse_args()
    
    # Run main SAPPHIRE pipeline
    if not args.robustness_only:
        print("\n" + "="*80)
        print("RUNNING SAPPHIRE MAIN PIPELINE")
        print("="*80 + "\n")
        
        from sapphire_pipeline import run_all_datasets, run_sapphire_pipeline
        from sapphire_pipeline import DATASETS, OUTPUT_DIR, SAPPHIRE_PARAMS
        
        if args.single_dataset:
            if args.single_dataset in DATASETS:
                results = run_sapphire_pipeline(
                    dataset_name=args.single_dataset,
                    h5ad_path=DATASETS[args.single_dataset],
                    output_dir=OUTPUT_DIR,
                    params=SAPPHIRE_PARAMS
                )
                
                # Plot
                from sapphire_pipeline import plot_sapphire_results
                plot_path = OUTPUT_DIR / args.single_dataset / f"{args.single_dataset}_sapphire_plot.pdf"
                plot_sapphire_results(results, save_path=plot_path)
            else:
                print(f"Error: Dataset '{args.single_dataset}' not found")
                print(f"Available: {list(DATASETS.keys())}")
                return
        else:
            results = run_all_datasets()
    
    # Run robustness checks
    if not args.sapphire_only:
        print("\n" + "="*80)
        print("RUNNING ROBUSTNESS CHECKS")
        print("="*80 + "\n")
        
        from robustness_pipeline import run_all_robustness_tests, run_robustness_pipeline
        from robustness_pipeline import TEST_DATASETS, ROBUSTNESS_OUTPUT, SAPPHIRE_PARAMS
        
        if args.single_dataset:
            if args.single_dataset in TEST_DATASETS:
                results = run_robustness_pipeline(
                    dataset_name=args.single_dataset,
                    h5ad_path=TEST_DATASETS[args.single_dataset],
                    output_dir=ROBUSTNESS_OUTPUT,
                    params=SAPPHIRE_PARAMS
                )
                
                # Plot
                from robustness_pipeline import plot_robustness_results
                plot_path = ROBUSTNESS_OUTPUT / args.single_dataset / f"{args.single_dataset}_robustness.pdf"
                plot_robustness_results(ROBUSTNESS_OUTPUT, args.single_dataset, save_path=plot_path)
            else:
                print(f"Error: Dataset '{args.single_dataset}' not in robustness test set")
                print(f"Available: {list(TEST_DATASETS.keys())}")
                return
        else:
            results = run_all_robustness_tests()
    
    print("\n" + "="*80)
    print("ALL PIPELINES COMPLETE!")
    print("="*80)
    print("\nCheck outputs in:")
    if not args.robustness_only:
        from sapphire_pipeline import OUTPUT_DIR
        print(f"  - Main results: {OUTPUT_DIR}")
    if not args.sapphire_only:
        from robustness_pipeline import ROBUSTNESS_OUTPUT
        print(f"  - Robustness: {ROBUSTNESS_OUTPUT}")


if __name__ == "__main__":
    main()
