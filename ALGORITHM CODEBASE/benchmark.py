import time
import csv
from typing import List, Dict, Any
from problem import ProblemInstance
from evaluator import TimetableEvaluator
from solvers import StandardARO, EnhancedARO, GA, SA, PSO

def run_experiment(problem: ProblemInstance, solver_class, solver_name: str,
                   runs: int = 30, max_evals: int = 100000, **solver_kwargs):
    evaluator = TimetableEvaluator(problem)
    results = []
    
    # Adjust evals for large problems - balance feasibility vs runtime
    adjusted_evals = max_evals
    if problem.num_events >= 400:
        adjusted_evals = 5000  # Increased from 1000 for better feasibility
    elif problem.num_events >= 200:
        adjusted_evals = 10000  # Increased from 5000
    elif problem.num_events >= 100:
        adjusted_evals = 15000  # Increased from 10000

    for run in range(runs):
        seed = 1000 + run
        solver = solver_class(max_evals=adjusted_evals, **solver_kwargs)
        start = time.time()
        best_tt, metrics = solver.solve(problem, evaluator, seed=seed)
        elapsed = time.time() - start
        
        # Progress output every run
        print(f"    Run {run + 1:2d}/{runs}: fitness={best_tt.fitness:12.1f} hard={best_tt.hard_violations:2d} time={elapsed:6.1f}s")

        results.append({
            'solver': solver_name,
            'instance': problem.name,
            'run': run + 1,
            'seed': seed,
            'best_fitness': metrics['best_fitness'],
            'hard_violations': metrics['hard_violations'],
            'soft_penalty': metrics['soft_penalty'],
            'evaluations': metrics['evaluations'],
            'time_sec': round(elapsed, 2),
            'feasible': 1 if metrics['hard_violations'] == 0 else 0
        })

    return results

def summarize(results: List[Dict[str, Any]]):
    from statistics import mean, stdev
    by_solver = {}
    for r in results:
        by_solver.setdefault(r['solver'], []).append(r)

    print(f"{'Solver':<20} {'Best':<10} {'Mean':<10} {'Std':<10} {'Feas%':<8} {'Time(s)':<10}")
    print("-" * 70)
    for solver, rows in by_solver.items():
        fits = [r['best_fitness'] for r in rows]
        times = [r['time_sec'] for r in rows]
        feas = sum(r['feasible'] for r in rows) / len(rows) * 100
        fit_std = stdev(fits) if len(fits) > 1 else 0.0
        print(f"{solver:<20} {min(fits):<10.1f} {mean(fits):<10.1f} {fit_std:<10.1f} {feas:<8.1f} {mean(times):<10.2f}")

def save_csv(results: List[Dict[str, Any]], filename: str = "results.csv"):
    if not results:
        return
    keys = results[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved raw results to {filename}")
