import sys
import os
import glob

# Add parent to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_itc2007
from benchmark import run_experiment, summarize, save_csv
from solvers import StandardARO, EnhancedARO, GA, SA, PSO

def quick_test():
    # Dataset configuration
    dataset_path = r"C:\Users\user\Videos\FINAL YEAR PROJECT\DATASETS\ITC_2007\Early data set"
    
    # Get all .tim files
    tim_files = sorted(glob.glob(os.path.join(dataset_path, "*.tim")))
    if not tim_files:
        print(f"ERROR: No .tim files found in {dataset_path}")
        return False
    
    # Test with first instance only
    tim_file = tim_files[0]
    instance_name = os.path.basename(tim_file)
    print(f"Quick validation test with: {instance_name}")
    print("="*70)
    
    try:
        problem = parse_itc2007(tim_file)
        print(f"OK Parse success: {problem.num_events} events, {len(problem.rooms)} rooms, {len(problem.courses)} courses")
    except Exception as e:
        print(f"FAIL Parse ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    all_results = []
    
    # Quick test: 1 run, 5000 evals per solver
    test_runs = 1
    test_evals = 5000
    
    try:
        print(f"\n  Testing StandardARO ({test_runs} run × {test_evals} evals)...")
        res = run_experiment(problem, StandardARO, "Standard_ARO", runs=test_runs, max_evals=test_evals)
        all_results.extend(res)
        print(f"  OK StandardARO completed")
    except Exception as e:
        print(f"  FAIL StandardARO ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print(f"  Testing EnhancedARO ({test_runs} run × {test_evals} evals)...")
        res = run_experiment(problem, EnhancedARO, "Enhanced_ARO", runs=test_runs, max_evals=test_evals)
        all_results.extend(res)
        print(f"  OK EnhancedARO completed")
    except Exception as e:
        print(f"  FAIL EnhancedARO ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print(f"  Testing GA ({test_runs} run × {test_evals} evals)...")
        res = run_experiment(problem, GA, "GA", runs=test_runs, max_evals=test_evals)
        all_results.extend(res)
        print(f"  OK GA completed")
    except Exception as e:
        print(f"  FAIL GA ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print(f"  Testing SA ({test_runs} run × {test_evals} evals)...")
        res = run_experiment(problem, SA, "SA", runs=test_runs, max_evals=test_evals)
        all_results.extend(res)
        print(f"  OK SA completed")
    except Exception as e:
        print(f"  FAIL SA ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print(f"  Testing PSO ({test_runs} run × {test_evals} evals)...")
        res = run_experiment(problem, PSO, "PSO", runs=test_runs, max_evals=test_evals)
        all_results.extend(res)
        print(f"  OK PSO completed")
    except Exception as e:
        print(f"  FAIL PSO ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test summary
    print("\n" + "="*70)
    print("VALIDATION TEST RESULTS")
    print("="*70)
    try:
        summarize(all_results)
    except Exception as e:
        print(f"ERROR in summarize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        save_csv(all_results, "test_results.csv")
    except Exception as e:
        print(f"ERROR saving CSV: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nOK All validation checks passed!")
    return True

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
