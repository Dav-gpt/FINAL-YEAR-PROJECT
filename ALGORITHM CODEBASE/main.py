import sys
import os
import glob
import time

# Add parent to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_itc2007
from benchmark import run_experiment, summarize, save_csv
from solvers import StandardARO, EnhancedARO, GA, SA, PSO

def main():
    # Dataset configuration
    dataset_path = r"C:\Users\user\Videos\FINAL YEAR PROJECT\DATASETS\ITC_2007\Early data set"
    
    # Get all .tim files in the Early dataset
    tim_files = sorted(glob.glob(os.path.join(dataset_path, "*.tim")))
    if not tim_files:
        print(f"ERROR: No .tim files found in {dataset_path}")
        return
    
    print(f"Found {len(tim_files)} problem instances in Early dataset")
    print("="*70)
    
    all_results = []
    overall_start = time.time()
    
    # Enhanced parameters for better feasibility
    base_runs = 15  # Increased from 10
    base_evals = 5000  # Increased from 1000
    
    # Process each problem instance
    for idx, tim_file in enumerate(tim_files, 1):
        instance_name = os.path.basename(tim_file)
        print(f"\n[{idx}/{len(tim_files)}] Processing: {instance_name}")
        print("-"*70)
        
        try:
            problem = parse_itc2007(tim_file)
            print(f"  Parsed successfully: {problem.num_events} events, {len(problem.rooms)} rooms, {len(problem.courses)} courses")
        except Exception as e:
            print(f"  ERROR parsing {instance_name}: {e}")
            continue
        
        instance_start = time.time()
        
        # Run each algorithm in optimal order (fastest first for earlier completion)
        print(f"\n  === Standard ARO ({base_runs} runs × {base_evals} evals) ===")
        res = run_experiment(problem, StandardARO, "Standard_ARO", runs=base_runs, max_evals=base_evals)
        all_results.extend(res)
        
        print(f"  === Genetic Algorithm ===")
        res = run_experiment(problem, GA, "GA", runs=base_runs, max_evals=base_evals)
        all_results.extend(res)
        
        print(f"  === Simulated Annealing ===")
        res = run_experiment(problem, SA, "SA", runs=base_runs, max_evals=base_evals)
        all_results.extend(res)
        
        print(f"  === Particle Swarm Optimization ===")
        res = run_experiment(problem, PSO, "PSO", runs=base_runs, max_evals=base_evals)
        all_results.extend(res)
        
        print(f"  === Enhanced ARO (optimized for large problems) ===")
        res = run_experiment(problem, EnhancedARO, "Enhanced_ARO", runs=base_runs, max_evals=base_evals)
        all_results.extend(res)
        
        instance_elapsed = time.time() - instance_start
        print(f"\n  Instance time: {instance_elapsed:.1f}s")
    
    overall_elapsed = time.time() - overall_start
    
    # Final summary
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS SUMMARY - EARLY DATASET")
    print("="*70)
    summarize(all_results)
    
    output_csv = "uctp_results_early_dataset.csv"
    save_csv(all_results, output_csv)
    
    print("\n" + "="*70)
    print(f"Total analysis time: {overall_elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_csv}")
    print(f"Total result entries: {len(all_results)}")
    print("="*70)

if __name__ == "__main__":
    main()
