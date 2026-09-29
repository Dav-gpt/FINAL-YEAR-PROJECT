"""
Comprehensive analysis of UCTP solver comparative results
Generates statistical summaries, visualizations, and detailed reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_results(csv_file):
    """Load and validate results CSV"""
    df = pd.read_csv(csv_file)
    print(f"OK Loaded {len(df)} results from {csv_file}\n")
    return df

def summary_by_solver(df):
    """Statistical summary grouped by solver"""
    print("="*80)
    print("SOLVER PERFORMANCE SUMMARY")
    print("="*80)
    
    summary = df.groupby('solver').agg({
        'best_fitness': ['min', 'mean', 'std', 'max'],
        'hard_violations': ['min', 'mean', 'max'],
        'soft_penalty': 'mean',
        'time_sec': 'mean',
        'feasible': 'sum'
    }).round(2)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    print(summary)
    print()

def summary_by_instance(df):
    """Summary grouped by problem instance"""
    print("="*80)
    print("INSTANCE DIFFICULTY ANALYSIS")
    print("="*80)
    
    instances = df.groupby('instance').agg({
        'best_fitness': 'mean',
        'hard_violations': 'mean',
        'feasible': 'sum',
    }).round(2)
    
    instances.columns = ['Avg_Fitness', 'Avg_Hard_Violations', 'Feasible_Count']
    print(instances)
    print()

def solver_comparison(df):
    """Detailed solver-by-instance matrix"""
    print("="*80)
    print("SOLVER × INSTANCE PERFORMANCE MATRIX (Mean Fitness)")
    print("="*80)
    
    matrix = df.pivot_table(
        values='best_fitness', 
        index='solver', 
        columns='instance', 
        aggfunc='mean'
    ).round(0)
    
    print(matrix)
    print(f"\nRow means (avg across instances):\n{matrix.mean(axis=1).round(0)}")
    print()

def statistical_tests(df):
    """ANOVA and pairwise comparisons"""
    print("="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    # ANOVA
    solvers = df['solver'].unique()
    groups = [df[df['solver'] == solver]['best_fitness'].values for solver in solvers]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"One-way ANOVA (all solvers):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Result: {'Significant differences' if p_value < 0.05 else 'No significant differences'}\n")
    
    # Pairwise Mann-Whitney U tests (non-parametric)
    print("Pairwise Mann-Whitney U tests (p-values):")
    solver_list = sorted(df['solver'].unique())
    for i, s1 in enumerate(solver_list):
        for s2 in solver_list[i+1:]:
            stat, p = stats.mannwhitneyu(
                df[df['solver'] == s1]['best_fitness'],
                df[df['solver'] == s2]['best_fitness']
            )
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {s1:15s} vs {s2:15s}: {p:.6f} {marker}")
    print()

def feasibility_analysis(df):
    """Analyze solution feasibility"""
    print("="*80)
    print("FEASIBILITY ANALYSIS")
    print("="*80)
    
    total = len(df)
    feasible = (df['feasible'] == 1).sum()
    
    print(f"Total solutions: {total}")
    print(f"Feasible (hard_violations=0): {feasible} ({100*feasible/total:.1f}%)")
    print(f"Infeasible (hard_violations>0): {total-feasible} ({100*(total-feasible)/total:.1f}%)\n")
    
    print("Feasibility by solver:")
    grouped = df.groupby('solver')['feasible']
    for solver, values in grouped:
        count = (values == 1).sum()
        total_runs = len(values)
        print(f"  {solver:15s}: {count:3d}/{total_runs} runs = {100*count/total_runs:5.1f}%")
    print()

def best_solutions(df, top_n=5):
    """Identify best solutions found"""
    print("="*80)
    print(f"TOP {top_n} SOLUTIONS OVERALL")
    print("="*80)
    
    best = df.nsmallest(top_n, 'best_fitness')[
        ['solver', 'instance', 'run', 'best_fitness', 'hard_violations', 'time_sec']
    ]
    print(best.to_string())
    print()

def create_visualizations(df, output_prefix='analysis'):
    """Generate publication-quality plots"""
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='solver', y='best_fitness', ax=ax)
    ax.set_ylabel('Best Fitness (lower is better)')
    ax.set_xlabel('Algorithm')
    ax.set_title('Fitness Distribution by Algorithm')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_boxplot.png', dpi=300)
    print(f"OK Saved {output_prefix}_boxplot.png")
    plt.close()
    
    # 2. Violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x='solver', y='best_fitness', ax=ax)
    ax.set_ylabel('Best Fitness')
    ax.set_xlabel('Algorithm')
    ax.set_title('Fitness Distribution (Violin Plot)')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_violin.png', dpi=300)
    print(f"OK Saved {output_prefix}_violin.png")
    plt.close()
    
    # 3. Runtime comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='solver', y='time_sec', ax=ax)
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Algorithm')
    ax.set_title('Average Execution Time per Run')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_runtime.png', dpi=300)
    print(f"OK Saved {output_prefix}_runtime.png")
    plt.close()
    
    # 4. Fitness vs Runtime scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    for solver in df['solver'].unique():
        solver_data = df[df['solver'] == solver]
        ax.scatter(solver_data['time_sec'], solver_data['best_fitness'], 
                  label=solver, alpha=0.6, s=50)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Solution Quality vs Execution Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_tradeoff.png', dpi=300)
    print(f"OK Saved {output_prefix}_tradeoff.png")
    plt.close()
    
    # 5. Heatmap: solver × instance
    pivot = df.pivot_table(values='best_fitness', index='solver', 
                           columns='instance', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Best Fitness'})
    ax.set_title('Performance Heatmap: Solver × Instance')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_heatmap.png', dpi=300)
    print(f"OK Saved {output_prefix}_heatmap.png")
    plt.close()
    
    print()

def export_summary_table(df, output_file='summary_table.csv'):
    """Export formatted summary table"""
    summary = df.groupby('solver').agg({
        'best_fitness': ['min', 'mean', 'std'],
        'hard_violations': 'mean',
        'time_sec': 'mean',
        'feasible': lambda x: (x==1).sum()
    }).round(2)
    
    summary.columns = ['Best', 'Mean', 'Std', 'Avg_Hard', 'Avg_Time', 'Feasible']
    summary = summary.sort_values('Mean')
    summary.to_csv(output_file)
    print(f"OK Exported summary to {output_file}")

def main():
    csv_file = 'uctp_results_early_dataset.csv'
    
    try:
        df = load_results(csv_file)
    except FileNotFoundError:
        print(f"ERROR: {csv_file} not found")
        return
    
    # Run all analyses
    summary_by_solver(df)
    summary_by_instance(df)
    solver_comparison(df)
    statistical_tests(df)
    feasibility_analysis(df)
    best_solutions(df, top_n=5)
    
    # Generate visualizations
    create_visualizations(df, output_prefix='uctp_analysis')
    
    # Export summary
    export_summary_table(df, output_file='uctp_summary.csv')
    
    print("="*80)
    print("OK ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - uctp_analysis_boxplot.png")
    print("  - uctp_analysis_violin.png")
    print("  - uctp_analysis_runtime.png")
    print("  - uctp_analysis_tradeoff.png")
    print("  - uctp_analysis_heatmap.png")
    print("  - uctp_summary.csv")

if __name__ == '__main__':
    main()
