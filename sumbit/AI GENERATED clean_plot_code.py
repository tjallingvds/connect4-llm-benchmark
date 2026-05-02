import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# ============================================================
# Configuration and Setup
# ============================================================

def setup_plot_style():
    """
    Configure matplotlib style settings for consistent plots.
    Sets up:
      - seaborn-v0_8-paper style for clean appearance
      - serif font family at size 10
      - axis labels at size 11, titles at size 12
      - tick labels at size 9
      - legend font size 9
    """
    plt.style.use('seaborn-v0_8-paper')
    rc('font', family='serif', size=10)
    rc('axes', labelsize=11, titlesize=12)
    rc('xtick', labelsize=9)
    rc('ytick', labelsize=9)
    rc('legend', fontsize=9)


# ============================================================
# Data Preparation Helpers
# ============================================================

def calculate_overall_mean_ranks(all_metrics, results_list):
    """
    Calculate mean rank distances across all positions for each model.
    Iterates through all category results and positions to collect rank distances.
    Returns:
      - List of mean rank distances, one per model
      - If no ranks found for a model, returns 0 for that model
    """
    overall_mean_ranks = []
    for i, metrics in enumerate(all_metrics):
        result_dict = results_list[i]
        all_ranks = []
        for category_data in result_dict['category_results'].values():
            for position in category_data['positions']:
                all_ranks.append(position['rank_distance'])
        mean_rank = sum(all_ranks) / len(all_ranks) if len(all_ranks) > 0 else 0
        overall_mean_ranks.append(mean_rank)
    return overall_mean_ranks


def get_category_avg_performance(all_metrics, categories_list):
    """
    Get average performance for a group of categories.
    For each model, averages the accuracy across specified categories.
    Behavior:
      - Converts category accuracies to percentages (multiplies by 100)
      - Only includes categories that exist in the model's metrics
      - Returns 0 if no matching categories found for a model
    Returns:
      - List of average accuracies (as percentages), one per model
    """
    perf_data = []
    for metrics in all_metrics:
        cat_dict = dict(zip(metrics['categories'], metrics['category_accuracies']))
        accuracies = []
        for cat in categories_list:
            if cat in cat_dict:
                accuracies.append(cat_dict[cat] * 100)
        avg_acc = sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0
        perf_data.append(avg_acc)
    return perf_data


# ============================================================
# Individual Plot Functions
# ============================================================

def plot_overall_accuracy(ax, models, overall_accs, patterns):
    """
    Plot overall optimal move choice percentage as bar chart.
    Creates white bars with black edges, applies hatch patterns for differentiation.
    Features:
      - Y-axis limited to 0-100% range
      - Grid lines for easier reading
      - Percentage labels above each bar
      - Top and right spines removed for cleaner look
    """
    bars = ax.bar(models, overall_accs, color='white', edgecolor='black', linewidth=1.5)
    for i, bar in enumerate(bars):
        bar.set_hatch(patterns[i])
    ax.set_ylabel('Optimal Move Choice (%)', fontweight='bold')
    ax.set_title('Overall Optimal Move Choice Percentage', fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{overall_accs[i]:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_overall_mean_distance(ax, models, overall_mean_ranks, patterns):
    """
    Plot overall mean distance from optimal move as bar chart.
    Lower values indicate better performance (closer to optimal moves).
    Features:
      - White bars with black edges and hatch patterns
      - Grid lines for easier reading
      - Value labels above each bar (2 decimal places)
      - "Lower is better" annotation in top-left corner
      - Top and right spines removed for cleaner look
    """
    bars = ax.bar(models, overall_mean_ranks, color='white', edgecolor='black', linewidth=1.5)
    for i, bar in enumerate(bars):
        bar.set_hatch(patterns[i])
    ax.set_ylabel('Mean Rank Distance', fontweight='bold')
    ax.set_title('Overall Mean Distance from Optimal Move', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{overall_mean_ranks[i]:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.02, 0.98, 'Lower is better', transform=ax.transAxes, 
             fontsize=8, verticalalignment='top', style='italic', color='gray')


def plot_category_accuracy(ax, all_metrics, categories, patterns):
    """
    Plot per category optimal move choice percentage as grouped bar chart.
    Each category gets a group of bars, one per model.
    Layout:
      - Bars offset by width * (i - 1.5) to center groups
      - Category labels rotated 45 degrees for readability
      - Legend positioned above plot with 2 columns
      - Y-axis limited to 0-100% range
    """
    x_pos = np.arange(len(categories))
    width = 0.2
    for i, metrics in enumerate(all_metrics):
        offset = width * (i - 1.5)
        accs = [acc * 100 for acc in metrics['category_accuracies']]
        bars = ax.bar(x_pos + offset, accs, width, label=metrics['model'], 
                color='white', edgecolor='black', linewidth=1.2, hatch=patterns[i])
    ax.set_ylabel('Optimal Move Choice (%)', fontweight='bold')
    ax.set_title('Per Category Optimal Move Choice Percentage', fontweight='bold', pad=35)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_category_mean_distance(ax, all_metrics, categories, patterns):
    """
    Plot per category mean distance from optimal move as grouped bar chart.
    Lower values indicate better performance within each category.
    Layout:
      - Bars offset by width * (i - 1.5) to center groups
      - Category labels rotated 45 degrees for readability
      - Legend positioned above plot with 2 columns
      - "Lower is better" annotation in top-left corner
    """
    x_pos = np.arange(len(categories))
    width = 0.2
    for i, metrics in enumerate(all_metrics):
        offset = width * (i - 1.5)
        avg_ranks = metrics['avg_rank_distances']
        bars = ax.bar(x_pos + offset, avg_ranks, width, label=metrics['model'],
                color='white', edgecolor='black', linewidth=1.2, hatch=patterns[i])
    ax.set_ylabel('Mean Rank Distance', fontweight='bold')
    ax.set_title('Per Category Mean Distance from Optimal Move', fontweight='bold', pad=35)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.02, 0.98, 'Lower is better', transform=ax.transAxes, 
             fontsize=8, verticalalignment='top', style='italic', color='gray')


def plot_rank_distribution_frequency(ax, all_metrics, results_list, markers, linestyles):
    """
    Plot distribution of move rankings as frequency histogram (line plot).
    Shows percentage of moves at each rank distance from optimal.
    Bins:
      - 0, 1, 2, 3, 4, and ≥5 (combines 5+)
    Each model gets a line with:
      - Unique marker style (circle, square, triangle, diamond)
      - Unique line style (solid, dashed, dash-dot, dotted)
      - White marker faces with black edges
    """
    rank_bins = [0, 1, 2, 3, 4, 5, 10]
    rank_labels = ['0', '1', '2', '3', '4', '≥5']
    
    for i, metrics in enumerate(all_metrics):
        result_dict = results_list[i]
        all_ranks = []
        for category_data in result_dict['category_results'].values():
            for position in category_data['positions']:
                all_ranks.append(position['rank_distance'])
        hist, _ = np.histogram(all_ranks, bins=rank_bins)
        hist = (hist / len(all_ranks)) * 100
        x_pos_hist = np.arange(len(rank_labels))
        ax.plot(x_pos_hist, hist, marker=markers[i], linewidth=2, label=metrics['model'], 
                 color='black', markersize=7, linestyle=linestyles[i], markerfacecolor='white', 
                 markeredgecolor='black', markeredgewidth=1.5)
    ax.set_xlabel('Rank Distance from Optimal', fontweight='bold')
    ax.set_ylabel('Frequency (%)', fontweight='bold')
    ax.set_title('Distribution of Move Rankings - Frequency', fontweight='bold', pad=35)
    ax.set_xticks(x_pos_hist)
    ax.set_xticklabels(rank_labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_rank_distribution_boxplot(ax, all_metrics, results_list, models, patterns):
    """
    Plot distribution of move rankings as box plot.
    Shows quartiles, median, and outliers for rank distances across all positions.
    Features:
      - White boxes with black edges and hatch patterns
      - Box width 0.6 for readability
      - Thick median line (width 2.5) for visibility
      - Model names as x-axis labels
    """
    rank_data = [[] for _ in range(len(all_metrics))]
    for i, metrics in enumerate(all_metrics):
        result_dict = results_list[i]
        for category_data in result_dict['category_results'].values():
            for position in category_data['positions']:
                rank_data[i].append(position['rank_distance'])
    bp = ax.boxplot(rank_data, tick_labels=models, patch_artist=True, widths=0.6,
                      boxprops=dict(linewidth=1.5, edgecolor='black'),
                      whiskerprops=dict(linewidth=1.2, color='black'),
                      capprops=dict(linewidth=1.2, color='black'),
                      medianprops=dict(linewidth=2.5, color='black'))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('white')
        patch.set_hatch(patterns[i])
    ax.set_ylabel('Rank Distance', fontweight='bold')
    ax.set_title('Distribution of Move Rankings - Box Plot', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_reasoning_complexity(ax, all_metrics, patterns):
    """
    Plot performance by reasoning complexity (grouped buckets).
    Categories grouped into three complexity levels:
      - Immediate Action: P1 Win in 1, P2 Win in 1
      - Short Reasoning: After 20 Moves, P1 Win in 3, P2 Win in 3
      - Long Reasoning: P1 Win in 5, P2 Win in 5, After 5 Moves
    Shows average accuracy across categories within each complexity group.
    Layout:
      - Bars offset by width * (i - 1.5) to center groups
      - Legend positioned above plot with 2 columns
      - Y-axis limited to 0-100% range
    """
    immediate_categories = ['P1 Win in 1', 'P2 Win in 1']
    medium_categories = ['After 20 Moves', 'P1 Win in 3', 'P2 Win in 3']
    long_categories = ['P1 Win in 5', 'P2 Win in 5', 'After 5 Moves']
    
    reasoning_groups = ['Immediate\nAction', 'Short\nReasoning', 'Long\nReasoning']
    x_pos_reason = np.arange(len(reasoning_groups))
    width_reason = 0.2
    
    immediate_perf = get_category_avg_performance(all_metrics, immediate_categories)
    medium_perf = get_category_avg_performance(all_metrics, medium_categories)
    long_perf = get_category_avg_performance(all_metrics, long_categories)
    
    for i, metrics in enumerate(all_metrics):
        offset_reason = width_reason * (i - 1.5)
        grouped_perf = [immediate_perf[i], medium_perf[i], long_perf[i]]
        ax.bar(x_pos_reason + offset_reason, grouped_perf, width_reason, label=metrics['model'],
                color='white', edgecolor='black', linewidth=1.2, hatch=patterns[i])
    
    ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
    ax.set_title('Performance by Reasoning Complexity', fontweight='bold', pad=35)
    ax.set_xticks(x_pos_reason)
    ax.set_xticklabels(reasoning_groups, fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================
# Master Plot Generation Function
# ============================================================

def generate_comprehensive_plot(all_metrics, human_results, openai_results, 
                                anthropic_results, deepseek_results, human_metrics,
                                output_filename='connect4_llm_evaluation.png'):
    """
    Generate comprehensive evaluation plot with all subplots in 3x3 grid.
    Creates 7 plots showing different aspects of model performance:
      - Plot 1: Overall optimal move choice percentage
      - Plot 2: Overall mean distance from optimal move
      - Plot 3: Per category optimal move choice percentage
      - Plot 4: Per category mean distance from optimal move
      - Plot 5: Distribution of move rankings (frequency)
      - Plot 6: Distribution of move rankings (box plot)
      - Plot 7: Performance by reasoning complexity
    Output:
      - Saves figure as PNG at 300 DPI
      - Uses tight layout with 2.5 padding
      - White background for clean appearance
    """
    setup_plot_style()
    
    # Shared data
    models = [m['model'] for m in all_metrics]
    results_list = [human_results, openai_results, anthropic_results, deepseek_results]
    patterns = ['', '///', 'xxx', '...']
    markers = ['o', 's', '^', 'd']
    linestyles = ['-', '--', '-.', ':']
    
    # Calculate overall metrics
    overall_accs = [m['overall_accuracy'] * 100 for m in all_metrics]
    overall_mean_ranks = calculate_overall_mean_ranks(all_metrics, results_list)
    categories = human_metrics['categories']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))
    
    # Plot 1: Overall Optimal Move Choice Percentage
    ax1 = plt.subplot(3, 3, 1)
    plot_overall_accuracy(ax1, models, overall_accs, patterns)
    
    # Plot 2: Overall Mean Distance from Optimal
    ax2 = plt.subplot(3, 3, 2)
    plot_overall_mean_distance(ax2, models, overall_mean_ranks, patterns)
    
    # Plot 3: Per Category Optimal Move Choice Percentage
    ax3 = plt.subplot(3, 3, 3)
    plot_category_accuracy(ax3, all_metrics, categories, patterns)
    
    # Plot 4: Per Category Mean Distance from Optimal
    ax4 = plt.subplot(3, 3, 4)
    plot_category_mean_distance(ax4, all_metrics, categories, patterns)
    
    # Plot 5: Distribution of Move Rankings - Frequency
    ax5 = plt.subplot(3, 3, 5)
    plot_rank_distribution_frequency(ax5, all_metrics, results_list, markers, linestyles)
    
    # Plot 6: Distribution of Move Rankings - Box Plot
    ax6 = plt.subplot(3, 3, 6)
    plot_rank_distribution_boxplot(ax6, all_metrics, results_list, models, patterns)
    
    # Plot 7: Performance by Reasoning Complexity
    ax7 = plt.subplot(3, 3, 7)
    plot_reasoning_complexity(ax7, all_metrics, patterns)
    
    # Finalize and save
    plt.tight_layout(pad=2.5)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# ============================================================
# Statistics Printing Function
# ============================================================

def print_performance_summary(all_metrics):
    """
    Print comprehensive performance analysis summary.
    For each model, displays:
      - Overall accuracy as percentage
      - Correct moves count (correct/total)
      - Average rank distance across all categories
    Output formatted with section headers and separators.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*80)
    for metrics in all_metrics:
        print(f"\n{metrics['model']}:")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
        print(f"  Correct Moves: {metrics['total_correct']}/{metrics['total_positions']}")
        avg_rank = sum(metrics['avg_rank_distances']) / len(metrics['avg_rank_distances'])
        print(f"  Average Rank Distance: {avg_rank:.2f}")
    print("\n" + "="*80)


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    # Note: This script expects the following variables to be defined:
    # - all_metrics
    # - human_results, openai_results, anthropic_results, deepseek_results
    # - human_metrics
    #
    # These should be loaded from plots_and_stats module or defined elsewhere
    
    try:
        from plots_and_stats import load_all_results
        
        # Load all results
        all_metrics, human_results, openai_results, anthropic_results, deepseek_results, human_metrics = load_all_results()
        
        # Generate comprehensive plot
        generate_comprehensive_plot(
            all_metrics, human_results, openai_results, 
            anthropic_results, deepseek_results, human_metrics
        )
        
        # Print performance summary
        print_performance_summary(all_metrics)
        
    except ImportError:
        print("Error: Could not import load_all_results from plots_and_stats")
        print("Please ensure plots_and_stats.py is in the same directory")
        print("\nAlternatively, define the required variables manually:")
        print("  - all_metrics")
        print("  - human_results, openai_results, anthropic_results, deepseek_results")
        print("  - human_metrics")
