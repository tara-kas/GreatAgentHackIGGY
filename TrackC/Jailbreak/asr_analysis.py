"""
Comprehensive ASR Analysis Runner

This script orchestrates the complete ASR analysis pipeline:
1. Dataset evaluation
2. ASR calculation
3. Statistical analysis
4. Visualization generation
5. Report generation
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from util_asr_analysis import ASRAnalyzer
from asr_visualization import ASRVisualizer


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "pandas", "numpy", "matplotlib", "seaborn", "scipy"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    return True


def check_data_files():
    """Check if required data files exist."""
    root_dir = Path(__file__).parent.parent.parent
    results_dir = root_dir / "TrackC" / "Jailbreak" / "hle-results"
    
    required_files = [
        results_dir / "hle_results_benign_main.csv",
        results_dir / "hle_results_harmful_main.csv",
        results_dir / "hle_results_jailbreak_main.csv"
    ]
    
    missing = []
    for file_path in required_files:
        if not file_path.exists():
            missing.append(str(file_path))
    
    if missing:
        print("WARNING: Some result files are missing:")
        for file_path in missing:
            print(f"  - {file_path}")
        print("\nRun asr_datasets.py first to generate these files.")
        return False
    
    return True


def run_full_pipeline(
    use_llm_judge: bool = False,
    max_judge_samples: int = 50,
    generate_visualizations: bool = True,
    attack_type: Optional[str] = None
):
    """
    Run the complete ASR analysis pipeline.
    
    Args:
        use_llm_judge: Whether to use LLM judge for re-evaluation
        max_judge_samples: Maximum samples to judge with LLM
        generate_visualizations: Whether to generate visualizations
        attack_type: Specific attack type to analyze (None for all)
    """
    print("\n" + "="*80)
    print("ASR ANALYSIS PIPELINE - COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return False
    
    # Check data files
    if not check_data_files():
        print("\nPlease generate result files first using asr_datasets.py")
        return False
    
    # Initialize analyzer
    print("\n[1/4] Initializing ASR Analyzer...")
    analyzer = ASRAnalyzer(use_llm_judge=use_llm_judge)
    
    # Run analysis
    print("\n[2/4] Running ASR Analysis...")
    if attack_type:
        analyzer.analyze_attack_type(
            attack_type,
            use_llm_judge=use_llm_judge,
            max_judge_samples=max_judge_samples
        )
    else:
        analyzer.run_full_analysis(
            use_llm_judge=use_llm_judge,
            max_judge_samples=max_judge_samples
        )
    
    # Generate reports
    print("\n[3/4] Generating Reports...")
    report_files = analyzer.generate_report()
    
    print("\nReport files generated:")
    for report_type, file_path in report_files.items():
        print(f"  ✓ {report_type}: {file_path}")
    
    # Generate visualizations
    if generate_visualizations:
        print("\n[4/4] Generating Visualizations...")
        try:
            visualizer = ASRVisualizer()
            visualizer.generate_all_visualizations()
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
            print("  (This is optional - reports are still available)")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive ASR Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis (heuristic judgment only)
  python run_asr_analysis.py

  # Run with LLM judge (slower but more accurate)
  python run_asr_analysis.py --use-llm-judge

  # Analyze specific attack type
  python run_asr_analysis.py --attack-type harmful

  # Skip visualizations
  python run_asr_analysis.py --no-viz
        """
    )
    
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM judge for re-evaluation (slower but more accurate)"
    )
    
    parser.add_argument(
        "--max-judge-samples",
        type=int,
        default=50,
        help="Maximum samples to judge with LLM (default: 50)"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    
    parser.add_argument(
        "--attack-type",
        choices=["benign", "harmful", "jailbreak"],
        default=None,
        help="Analyze specific attack type only (default: all)"
    )
    
    args = parser.parse_args()
    
    success = run_full_pipeline(
        use_llm_judge=args.use_llm_judge,
        max_judge_samples=args.max_judge_samples,
        generate_visualizations=not args.no_viz,
        attack_type=args.attack_type
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

