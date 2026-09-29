import os
import time
import uuid
import pandas as pd
import numpy as np
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Import UCTP solver modules
from parser import parse_file
from problem import ProblemInstance
from evaluator import TimetableEvaluator, Timetable
from solvers import StandardARO, EnhancedARO, SA

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['STATIC_PLOT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'plots')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create directories if they do not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_PLOT_FOLDER'], exist_ok=True)

class TrackingEvaluator(TimetableEvaluator):
    """Subclass of TimetableEvaluator that tracks the best fitness history per evaluation count"""
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

# Map of solver names to their classes
SOLVERS_MAP = {
    'Standard_ARO': StandardARO,
    'Enhanced_ARO': EnhancedARO,
    'SA': SA
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename_lower = file.filename.lower()
    if file and (filename_lower.endswith('.tim') or filename_lower.endswith('.json') or filename_lower.endswith('.csv')):
        filename = secure_filename(file.filename)
        # Generate a unique name to avoid conflicts
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            problem = parse_file(filepath)
            # Return basic info about the dataset
            return jsonify({
                'success': True,
                'filename': filename,
                'unique_filename': unique_filename,
                'events_count': problem.num_events,
                'rooms_count': len(problem.rooms),
                'courses_count': len(problem.courses),
                'students_count': len(problem.students),
                'curricula_count': len(problem.curricula)
            })
        except Exception as e:
            # Clean up file if parsing fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f"Failed to parse dataset file: {str(e)}"}), 400
            
    return jsonify({'error': 'Invalid file type. Only .tim, .json, and .csv files are allowed.'}), 400

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    data = request.json or {}
    unique_filename = data.get('unique_filename')
    selected_solvers = data.get('solvers', ['Standard_ARO', 'Enhanced_ARO', 'SA'])
    runs_count = int(data.get('runs', 3))
    eval_budget = int(data.get('budget', 5000))
    
    # Enforce evaluation budget constraints (5,000 to 15,000)
    eval_budget = max(5000, min(15000, eval_budget))
    
    base_seed = int(data.get('seed', 1000))
    
    if not unique_filename:
        return jsonify({'error': 'No file uploaded or unique_filename missing'}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Uploaded file not found on server'}), 404
        
    try:
        problem = parse_file(filepath)
    except Exception as e:
        return jsonify({'error': f"Failed to parse problem: {str(e)}"}), 400
        
    results = []
    best_overall_timetable = None
    best_overall_fitness = float('inf')
    best_overall_solver = ""
    
    # Store convergence histories for each algorithm
    convergence_histories = {}
    best_tt_per_solver = {}
    
    for solver_name in selected_solvers:
        if solver_name not in SOLVERS_MAP:
            continue
            
        solver_class = SOLVERS_MAP[solver_name]
        
        # Run convergence tracking on the first run of the algorithm
        evaluator_track = TrackingEvaluator(problem)
        solver_track = solver_class(max_evals=eval_budget)
        
        try:
            best_tt_track, metrics_track = solver_track.solve(problem, evaluator_track, seed=base_seed)
            # Filter history to prevent too many points in JSON (sample max 200 points)
            raw_hist = evaluator_track.history
            if len(raw_hist) > 200:
                indices = np.linspace(0, len(raw_hist) - 1, 200, dtype=int)
                sampled_hist = [raw_hist[idx] for idx in indices]
            else:
                sampled_hist = raw_hist
            convergence_histories[solver_name] = sampled_hist
        except Exception as e:
            print(f"Error tracking convergence for {solver_name}: {e}")
            convergence_histories[solver_name] = []
            
        # Run standard runs to get stats
        best_solver_timetable = None
        best_solver_fitness = float('inf')
        for run in range(runs_count):
            run_seed = base_seed + run
            evaluator = TimetableEvaluator(problem)
            solver_instance = solver_class(max_evals=eval_budget)
            
            start_time = time.time()
            try:
                best_tt, metrics = solver_instance.solve(problem, evaluator, seed=run_seed)
                elapsed = time.time() - start_time
                
                fitness = best_tt.fitness
                hard_violations = best_tt.hard_violations
                soft_penalty = best_tt.soft_penalty
                
                # Update best for this solver
                if fitness < best_solver_fitness:
                    best_solver_fitness = fitness
                    best_solver_timetable = best_tt.copy()
                    
                # Update best overall solution
                if fitness < best_overall_fitness:
                    best_overall_fitness = fitness
                    best_overall_timetable = best_tt
                    best_overall_solver = solver_name
                    
                results.append({
                    'solver': solver_name,
                    'instance': problem.name,
                    'run': run + 1,
                    'seed': run_seed,
                    'best_fitness': float(fitness),
                    'hard_violations': int(hard_violations),
                    'soft_penalty': int(soft_penalty),
                    'evaluations': int(metrics.get('evaluations', evaluator.evaluation_count)),
                    'time_sec': round(elapsed, 2),
                    'feasible': 1 if hard_violations == 0 else 0
                })
            except Exception as e:
                print(f"Error running solver {solver_name} in run {run+1}: {e}")
                
        if best_solver_timetable:
            best_tt_per_solver[solver_name] = best_solver_timetable
            
    if not results:
        return jsonify({'error': 'All solvers failed to execute.'}), 500
        
    df = pd.DataFrame(results)
    
    # Calculate stats grouped by solver
    summary_stats = {}
    for name in selected_solvers:
        solver_df = df[df['solver'] == name]
        if solver_df.empty:
            continue
            
        summary_stats[name] = {
            'best_fitness': float(solver_df['best_fitness'].min()),
            'mean_fitness': float(solver_df['best_fitness'].mean()),
            'std_fitness': float(solver_df['best_fitness'].std()) if len(solver_df) > 1 else 0.0,
            'mean_hard_violations': float(solver_df['hard_violations'].mean()),
            'mean_soft_penalty': float(solver_df['soft_penalty'].mean()),
            'mean_time_sec': float(solver_df['time_sec'].mean()),
            'feasibility_rate': float((solver_df['feasible'] == 1).sum() / len(solver_df) * 100)
        }
        
    # Generate static Matplotlib charts in static/plots/
    # 1. Box plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='solver', y='best_fitness')
    plt.ylabel('Best Fitness (Lower is Better)')
    plt.xlabel('Algorithm')
    plt.title('Fitness Distribution by Algorithm')
    plt.tight_layout()
    boxplot_path = os.path.join(app.config['STATIC_PLOT_FOLDER'], 'boxplot.png')
    plt.savefig(boxplot_path, dpi=200)
    plt.close()
    
    # 2. Runtime bar chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='solver', y='time_sec', errorbar=None)
    plt.ylabel('Execution Time (seconds)')
    plt.xlabel('Algorithm')
    plt.title('Average Execution Time per Run')
    plt.tight_layout()
    runtime_path = os.path.join(app.config['STATIC_PLOT_FOLDER'], 'runtime.png')
    plt.savefig(runtime_path, dpi=200)
    plt.close()
    
    # 3. Convergence curve
    plt.figure(figsize=(8, 5))
    colors = {
        'Standard_ARO': '#94a3b8',
        'Enhanced_ARO': '#06b6d4',
        'SA': '#6366f1'
    }
    for s_name, hist in convergence_histories.items():
        if not hist:
            continue
        evals, fits = zip(*hist)
        plt.step(evals, fits, label=s_name, color=colors.get(s_name, '#000000'), where='post', lw=2)
    plt.yscale('log')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Best Fitness Score (Log Scale)')
    plt.title('Algorithm Convergence Curve')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    convergence_path = os.path.join(app.config['STATIC_PLOT_FOLDER'], 'convergence.png')
    plt.savefig(convergence_path, dpi=200)
    plt.close()
    
    # 4. Solution quality vs time trade-off scatter
    plt.figure(figsize=(8, 5))
    for name in selected_solvers:
        solver_df = df[df['solver'] == name]
        if solver_df.empty:
            continue
        plt.scatter(solver_df['time_sec'], solver_df['best_fitness'], label=name, alpha=0.7, s=80)
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Best Fitness')
    plt.title('Solution Quality vs Execution Time Trade-off')
    plt.legend()
    plt.tight_layout()
    tradeoff_path = os.path.join(app.config['STATIC_PLOT_FOLDER'], 'tradeoff.png')
    plt.savefig(tradeoff_path, dpi=200)
    plt.close()

    # Formulate timetables per solver for rendering
    timetables_data = {}
    for s_name, tt in best_tt_per_solver.items():
        timetable_grid = []
        for eidx, (rid, period) in tt.assignments.items():
            cid = problem.event_course[eidx]
            timetable_grid.append({
                'event_index': int(eidx),
                'course': str(cid),
                'room': str(rid),
                'period': int(period),
                'day': int(period // problem.periods_per_day),
                'slot': int(period % problem.periods_per_day)
            })
        timetables_data[s_name] = timetable_grid
            
    # Generate text explanation
    explanation = generate_text_explanation(problem, summary_stats, best_overall_solver, best_overall_fitness)

    return jsonify({
        'success': True,
        'summary': summary_stats,
        'convergence_histories': convergence_histories,
        'best_solver': best_overall_solver,
        'best_fitness': float(best_overall_fitness),
        'timetables': timetables_data,
        'explanation': explanation
    })

def generate_text_explanation(problem, stats, best_solver, best_fitness) -> str:
    """Generate academic discussion text based on run results"""
    best_solver_display = best_solver.replace('_', ' ')
    
    exp = "### Comparative Analysis Discussion\n\n"
    exp += f"An empirical analysis was conducted on the uploaded dataset **{problem.name}**, consisting of **{problem.num_events} events**, **{len(problem.rooms)} rooms**, and **{len(problem.curricula)} curriculum configurations**.\n\n"
    
    exp += "#### Key Findings:\n"
    exp += f"- **Top Performer**: The best overall solution quality was found by **{best_solver_display}**, which achieved a best fitness score of **{best_fitness:,.0f}**. "
    
    if best_fitness < 1e6:
        exp += "This represents a **feasible timetable solution** (0 hard violations), satisfying all academic constraint specifications. "
    else:
        hard_violations = int(best_fitness // 1e6)
        soft_penalty = int(best_fitness % 1e6)
        exp += f"This solution remains infeasible, having **{hard_violations} hard constraint violations** and a soft penalty of **{soft_penalty:,.0f}**. "
        
    exp += "\n\n"
    
    # Analyze ARO comparison
    if 'Enhanced_ARO' in stats and 'Standard_ARO' in stats:
        enh_fit = stats['Enhanced_ARO']['best_fitness']
        std_fit = stats['Standard_ARO']['best_fitness']
        improvement = ((std_fit - enh_fit) / std_fit) * 100 if std_fit > 0 else 0
        exp += "#### Enhanced ARO vs. Standard ARO:\n"
        exp += f"Our proposed **Enhanced ARO** achieved a best fitness of **{enh_fit:,.0f}**, compared to **{std_fit:,.0f}** for **Standard ARO**. This represents a **{improvement:.1f}% improvement** in solution quality. "
        exp += "This performance leap highlights the critical benefit of incorporating domain-aware operations. Standard ARO relies on continuous positions mapped discretely, creating a discontinuous fitness landscape that limits guided search. Enhanced ARO's use of discrete operators (like Kempe chain swaps) directly resolves scheduling overlaps while preserving feasibility."
        exp += "\n\n"
        
    # Analyze other baselines
    exp += "#### Algorithmic Comparison & Resource Trade-offs:\n"
    if 'SA' in stats:
        exp += "1. **Simulated Annealing (SA)**: Because SA operates as a single-solution metaheuristic, it executes far more iterations than population-based methods under a fixed evaluation budget. This high iteration density provides a very effective local search. However, without population diversity, it is vulnerable to local minima on extremely constrained landscapes.\n"
        
    exp += "\n#### Conclusion:\n"
    exp += f"This experiment confirms that for the instance **{problem.name}**, **{best_solver_display}** is the recommended choice. If complete feasibility is required, the cooperative operators in Enhanced ARO (Greedy Repair combined with Kempe Chain Perturbation) provide a crucial capability for resolving complex scheduling conflicts."
    
    return exp

@app.route('/parser-ui')
def parser_ui():
    return render_template('parser_ui.html')

@app.route('/parse-details', methods=['POST'])
def parse_details():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename_lower = file.filename.lower()
    if file and (filename_lower.endswith('.tim') or filename_lower.endswith('.json') or filename_lower.endswith('.csv')):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            problem = parse_file(filepath)
            
            # Format rooms for UI
            rooms_list = []
            for rid, room in problem.rooms.items():
                rooms_list.append({
                    'id': str(rid),
                    'capacity': int(room.capacity)
                })
                
            # Format courses
            courses_list = []
            for cid, course in problem.courses.items():
                courses_list.append({
                    'id': str(cid),
                    'teacher': str(course.teacher),
                    'lectures': int(course.lectures),
                    'students': int(course.students)
                })
                
            # Format curricula
            curricula_list = []
            for curid, curriculum in problem.curricula.items():
                curricula_list.append({
                    'id': str(curid),
                    'courses': list(curriculum.courses)
                })
                
            # Format period constraints
            period_constraints_list = []
            for cid, banned in problem.period_constraints.items():
                period_constraints_list.append({
                    'course_id': str(cid),
                    'banned_periods': list(banned)
                })
                
            # Format room constraints
            room_constraints_list = []
            for cid, allowed in problem.room_constraints.items():
                room_constraints_list.append({
                    'course_id': str(cid),
                    'allowed_rooms': list(allowed)
                })
                
            # Clean up the file
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return jsonify({
                'success': True,
                'name': problem.name,
                'days': problem.days,
                'periods_per_day': problem.periods_per_day,
                'num_events': problem.num_events,
                'rooms': rooms_list,
                'courses': courses_list,
                'students_count': len(problem.students),
                'curricula': curricula_list,
                'period_constraints': period_constraints_list,
                'room_constraints': room_constraints_list
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f"Failed to parse file: {str(e)}"}), 400
            
    return jsonify({'error': 'Invalid file type. Only .tim, .json, and .csv files are allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
