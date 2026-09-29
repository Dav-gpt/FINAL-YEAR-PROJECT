import sys
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Ensure project dir is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_itc2007
from solvers import StandardARO, EnhancedARO, GA, SA, PSO
from evaluator import TimetableEvaluator, Timetable

class TrackingEvaluator(TimetableEvaluator):
    def __init__(self, problem, hard_weight=1e6):
        super().__init__(problem, hard_weight)
        self.history = []  # List of tuples (eval_count, best_fitness)
        self.best_fitness = float('inf')

    def reset_count(self):
        super().reset_count()
        self.history = []
        self.best_fitness = float('inf')

    def evaluate(self, timetable: Timetable):
        is_new = timetable.fitness is None
        fitness, hard, soft = super().evaluate(timetable)
        if is_new:
            if fitness < self.best_fitness:
                self.best_fitness = fitness
            self.history.append((self.evaluation_count, self.best_fitness))
        return fitness, hard, soft

def generate_convergence_plot():
    dataset_path = r"C:\Users\user\Videos\FINAL YEAR PROJECT\DATASETS\ITC_2007\Early data set"
    tim_files = sorted(glob.glob(os.path.join(dataset_path, "*.tim")))
    if not tim_files:
        print(f"Error: No .tim files found in {dataset_path}")
        return

    tim_file = tim_files[0]
    print(f"Running convergence test on: {os.path.basename(tim_file)}")
    problem = parse_itc2007(tim_file)

    solvers = {
        "Standard_ARO": (StandardARO, {"pop_size": 20}),
        "Enhanced_ARO": (EnhancedARO, {"pop_size": 20}),
        "GA": (GA, {"pop_size": 20}),
        "SA": (SA, {}),
        "PSO": (PSO, {"pop_size": 20})
    }

    max_evals = 5000
    seed = 42

    plt.figure(figsize=(10, 6), dpi=300)
    
    # Custom color palette matching the dashboard design
    colors = {
        "Standard_ARO": "#94a3b8",   # Slate
        "Enhanced_ARO": "#06b6d4",   # Cyan
        "GA": "#f59e0b",             # Amber
        "SA": "#6366f1",             # Indigo (Primary)
        "PSO": "#ec4899"             # Pink
    }

    for name, (solver_class, kwargs) in solvers.items():
        print(f"Running {name}...")
        evaluator = TrackingEvaluator(problem)
        solver = solver_class(max_evals=max_evals, **kwargs)
        
        # Run solver
        try:
            solver.solve(problem, evaluator, seed=seed)
        except Exception as e:
            print(f"Error running {name}: {e}")
            continue

        # Process history
        history = evaluator.history
        if not history:
            continue
        
        evals, fitness = zip(*history)
        
        # Map to arrays and filter/interpolate to have smooth curves
        evals = np.array(evals)
        fitness = np.array(fitness)
        
        # Plot step function for convergence
        plt.step(evals, fitness, label=name, color=colors.get(name, "#000000"), where='post', lw=2)

    plt.yscale('log')
    plt.xlabel('Number of Function Evaluations', fontsize=11, fontweight='bold')
    plt.ylabel('Best Fitness Score (Log Scale)', fontsize=11, fontweight='bold')
    plt.title('Algorithm Convergence Curve on comp-2007-2-1', fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    
    output_path = "uctp_analysis_convergence.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print("Flowchart image saved successfully as uctp_analysis_convergence.png")

if __name__ == "__main__":
    generate_convergence_plot()
