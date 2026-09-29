"""
Extended comparative analysis across all ITC-2007 Track 2 dataset tracks
Runs analysis on Early, Late, and Hidden datasets
"""

import sys
import os
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_itc2007
from benchmark import run_experiment, summarize, save_csv
from solvers import StandardARO, EnhancedARO, GA, SA, PSO

def run_track(track_name, dataset_path, base_runs=15, base_evals=5000):
    """Run comparative analysis on entire track"""
    
    tim_files = sorted(glob.glob(os.path.join(dataset_path, "*.tim")))
    if not tim_files:
        print(f"ERROR: No .tim files found in {dataset_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {track_name.upper()} DATASET")
    print(f"{'='*70}")
    print(f"Found {len(tim_files)} problem instances")
    print()
    
    all_results = []
    track_start = time.time()
    
    # Process each instance
    for idx, tim_file in enumerate(tim_files, 1):
        instance_name = os.path.basename(tim_file)
        print(f"[{idx}/{len(tim_files)}] Processing: {instance_name}")
        print("-"*70)
        
        try:
            problem = parse_itc2007(tim_file)
            print(f"  Parsed: {problem.num_events} events, {len(problem.rooms)} rooms, {len(problem.courses)} courses")
        except Exception as e:
            print(f"  ERROR parsing: {e}")
            continue
        
        instance_start = time.time()
        
        # Solvers in optimal order (fastest first)
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
        
        print(f"  === Enhanced ARO (optimized) ===")
        res = run_experiment(problem, EnhancedARO, "Enhanced_ARO", runs=base_runs, max_evals=base_evals)
        all_results.extend(res)
        
        instance_elapsed = time.time() - instance_start
        print(f"\n  Instance time: {instance_elapsed:.1f}s\n")
    
    track_elapsed = time.time() - track_start
    
    # Save results for this track
    output_csv = f"uctp_results_{track_name}.csv"
    save_csv(all_results, output_csv)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY - {track_name.upper()}")
    print(f"{'='*70}")
    summarize(all_results)
    print(f"\nTrack time: {track_elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_csv}")
    print(f"Total entries: {len(all_results)}\n")
    
    return all_results, output_csv

def main():
    dataset_base = r"C:\Users\user\Videos\FINAL YEAR PROJECT\DATASETS\ITC_2007"
    
    # Define tracks
    tracks = {
        'early': os.path.join(dataset_base, "Early data set"),
        'late': os.path.join(dataset_base, "Late data set"),
        'hidden': os.path.join(dataset_base, "Hidden data set"),
    }
    
    overall_start = time.time()
    all_track_results = {}
    
    print("\n" + "="*70)
    print("ITC-2007 TRACK 2: COMPREHENSIVE COMPARATIVE ANALYSIS")
    print("="*70)
    
    # Run each track
    for track_name, dataset_path in tracks.items():
        if os.path.exists(dataset_path):
            result = run_track(track_name, dataset_path, base_runs=15, base_evals=5000)
            if result:
                all_track_results[track_name] = result
        else:
            print(f"WARNING: Track '{track_name}' not found at {dataset_path}")
    
    overall_elapsed = time.time() - overall_start
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL TRACKS")
    print("="*70)
    
    total_entries = 0
    for track_name, (results, csv_file) in all_track_results.items():
        total_entries += len(results)
        print(f"OK {track_name:10s}: {len(results):4d} entries -> {csv_file}")
    
    print(f"\nTotal analysis time: {overall_elapsed/60:.1f} minutes")
    print(f"Total result entries: {total_entries}")
    print(f"Tracks completed: {len(all_track_results)}/{len(tracks)}")
    print("="*70)

if __name__ == "__main__":
    main()
