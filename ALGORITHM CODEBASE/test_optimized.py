import sys
import os

# Add parent to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_itc2007
from benchmark import run_experiment
from solvers import StandardARO, GA, SA, PSO, EnhancedARO

def quick_validation():
    # Load first instance
    dataset_path = r"C:\Users\user\Videos\FINAL YEAR PROJECT\DATASETS\ITC_2007\Early data set"
    tim_file = os.path.join(dataset_path, "comp-2007-2-1 (1).tim")
    
    print("=" * 70)
    print("QUICK VALIDATION TEST - Optimized Settings")
    print("=" * 70)
    
    try:
        problem = parse_itc2007(tim_file)
        print(f"OK Parsed: {problem.num_events} events, {len(problem.rooms)} rooms")
    except Exception as e:
        print(f"FAIL Parse error: {e}")
        return False
    
    # Test with REALISTIC settings for 400-event problem
    test_runs = 2
    test_evals = 5000
    
    print(f"\nTesting with {test_runs} runs × {test_evals} evals (adaptive):\n")
    
    all_results = []
    
    # Test each solver
    solvers = [
        (StandardARO, "Standard ARO"),
        (GA, "Genetic Algorithm"),
        (SA, "Simulated Annealing"),
        (PSO, "Particle Swarm"),
        (EnhancedARO, "Enhanced ARO"),
    ]
    
    for solver_class, solver_name in solvers:
        try:
            print(f"  {solver_name:20s} ", end="", flush=True)
            res = run_experiment(problem, solver_class, solver_name, runs=test_runs, max_evals=test_evals)
            all_results.extend(res)
            print("OK Complete")
        except Exception as e:
            print(f"FAIL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 70)
    print(f"OK SUCCESS: All {len(solvers)} solvers completed without hanging")
    print(f"OK Collected {len(all_results)} result entries")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = quick_validation()
    sys.exit(0 if success else 1)
