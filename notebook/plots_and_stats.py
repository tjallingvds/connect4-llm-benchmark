import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


def load_results(filename):
    """
    Load results from JSON file
    
    Args:
        filename: path to JSON file
    
    Returns:
        dict with results
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def load_and_extract_metrics(filename, model_name):
    """
    Load results from JSON file and extract metrics in one operation
    
    Args:
        filename: path to JSON file
        model_name: name of the model
    
    Returns:
        tuple of (results_dict, metrics_dict)
    """
    results_dict = load_results(filename)
    
    metrics = {
        'model': model_name,
        'overall_accuracy': results_dict['overall_accuracy'],
        'total_correct': results_dict['total_correct'],
        'total_positions': results_dict['total_positions']
    }
    
    categories = []
    accuracies = []
    avg_rank_distances = []
    
    for category_name, category_data in results_dict['category_results'].items():
        categories.append(category_name)
        accuracies.append(category_data['accuracy'])
        
        rank_distances = []
        for position in category_data['positions']:
            rank_distances.append(position['rank_distance'])
        
        if len(rank_distances) > 0:
            avg_rank_dist = sum(rank_distances) / len(rank_distances)
        else:
            avg_rank_dist = 0
        avg_rank_distances.append(avg_rank_dist)
    
    metrics['categories'] = categories
    metrics['category_accuracies'] = accuracies
    metrics['avg_rank_distances'] = avg_rank_distances
    
    return results_dict, metrics


def load_all_results():
    """
    Load all result files and extract metrics
    
    Returns:
        tuple of (all_metrics, human_results, openai_results, anthropic_results, deepseek_results, human_metrics)
    """
    openai_results, openai_metrics = load_and_extract_metrics('openai_results2.json', 'GPT-4o')
    anthropic_results, anthropic_metrics = load_and_extract_metrics('anthropic_results.json', 'Claude Sonnet 4.5')
    deepseek_results, deepseek_metrics = load_and_extract_metrics('deepseek_results.json', 'DeepSeek')
    human_results, human_metrics = load_and_extract_metrics('human_results.json', 'Human Expert')
    
    all_metrics = [human_metrics, openai_metrics, anthropic_metrics, deepseek_metrics]
    
    return all_metrics, human_results, openai_results, anthropic_results, deepseek_results, human_metrics


def generate_all_plots(all_metrics, human_results, openai_results, anthropic_results, deepseek_results):
    """
    Generate all plots separately with consistent dimensions and styling
    
    Args:
        all_metrics: List of metric dictionaries for each model
        human_results: Results dictionary for human expert
        openai_results: Results dictionary for GPT-4o
        anthropic_results: Results dictionary for Claude Sonnet 4.5
        deepseek_results: Results dictionary for DeepSeek
    """
    plt.style.use('seaborn-v0_8-paper')
    rc('font', family='serif', size=10)
    rc('axes', labelsize=11, titlesize=12)
    rc('xtick', labelsize=9)
    rc('ytick', labelsize=9)
    rc('legend', fontsize=9)

    # Shared data
    models = [m['model'] for m in all_metrics]
    results_list = [human_results, openai_results, anthropic_results, deepseek_results]
    patterns = ['', '///', 'xxx', '...']
    markers = ['o', 's', '^', 'd']
    linestyles = ['-', '--', '-.', ':']

    # Calculate overall metrics
    overall_accs = [m['overall_accuracy'] * 100 for m in all_metrics]

    overall_mean_ranks = []
    for i, metrics in enumerate(all_metrics):
        result_dict = results_list[i]
        all_ranks = []
        for category_data in result_dict['category_results'].values():
            for position in category_data['positions']:
                all_ranks.append(position['rank_distance'])
        mean_rank = sum(all_ranks) / len(all_ranks) if len(all_ranks) > 0 else 0
        overall_mean_ranks.append(mean_rank)

    # Plot 1: Overall Optimal Move Choice Percentage
    fig, ax = plt.subplots(figsize=(8, 6))
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
    plt.tight_layout()
    plt.savefig('plot1_overall_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Plot 2: Overall Mean Distance from Optimal Move
    fig, ax = plt.subplots(figsize=(8, 6))
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
    plt.tight_layout()
    plt.savefig('plot2_overall_mean_distance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Plot 3: Per Category Optimal Move Choice Percentage
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = all_metrics[0]['categories']
    x_pos = np.arange(len(categories))
    width = 0.2
    for i, metrics in enumerate(all_metrics):
        offset = width * (i - 1.5)
        accs = [acc * 100 for acc in metrics['category_accuracies']]
        bars = ax.bar(x_pos + offset, accs, width, label=metrics['model'], 
                color='white', edgecolor='black', linewidth=1.2, hatch=patterns[i])
    ax.set_ylabel('Optimal Move Choice (%)', fontweight='bold')
    ax.set_title('Per Category Optimal Move Choice Percentage', fontweight='bold', pad=50)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('plot3_category_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Plot 4: Per Category Mean Distance from Optimal Move
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, metrics in enumerate(all_metrics):
        offset = width * (i - 1.5)
        avg_ranks = metrics['avg_rank_distances']
        bars = ax.bar(x_pos + offset, avg_ranks, width, label=metrics['model'],
                color='white', edgecolor='black', linewidth=1.2, hatch=patterns[i])
    ax.set_ylabel('Mean Rank Distance', fontweight='bold')
    ax.set_title('Per Category Mean Distance from Optimal Move', fontweight='bold', pad=50)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.02, 0.98, 'Lower is better', transform=ax.transAxes, 
             fontsize=8, verticalalignment='top', style='italic', color='gray')
    plt.tight_layout()
    plt.savefig('plot4_category_mean_distance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Plot 5: Distribution of Move Rankings - Frequency
    rank_bins = [0, 1, 2, 3, 4, 5, 10]
    rank_labels = ['0', '1', '2', '3', '4', '≥5']

    fig, ax = plt.subplots(figsize=(8, 6))
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
    ax.set_title('Distribution of Move Rankings - Frequency', fontweight='bold', pad=50)
    ax.set_xticks(x_pos_hist)
    ax.set_xticklabels(rank_labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('plot5_rank_distribution_frequency.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Plot 6: Distribution of Move Rankings - Box Plot
    fig, ax = plt.subplots(figsize=(8, 6))
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
    plt.tight_layout()
    plt.savefig('plot6_rank_distribution_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Plot 7: Performance by Reasoning Complexity
    immediate_categories = ['P1 Win in 1', 'P2 Win in 1 (Block)']
    short_categories = ['Late Game', 'P1 Win in 3', 'P2 Win in 3']
    long_categories = ['P1 Win in 5', 'P2 Win in 5', 'Early Game']

    def get_category_avg_performance(categories_list):
        """Get average performance for a group of categories"""
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

    fig, ax = plt.subplots(figsize=(8, 6))
    reasoning_groups = ['Immediate\nAction', 'Short\nPlanning', 'Long\nPlanning']
    x_pos_reason = np.arange(len(reasoning_groups))
    width_reason = 0.2

    immediate_perf = get_category_avg_performance(immediate_categories)
    short_perf = get_category_avg_performance(short_categories)
    long_perf = get_category_avg_performance(long_categories)

    for i, metrics in enumerate(all_metrics):
        offset_reason = width_reason * (i - 1.5)
        grouped_perf = [immediate_perf[i], short_perf[i], long_perf[i]]
        ax.bar(x_pos_reason + offset_reason, grouped_perf, width_reason, label=metrics['model'],
                color='white', edgecolor='black', linewidth=1.2, hatch=patterns[i])

    ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
    ax.set_title('Performance by Reasoning Complexity', fontweight='bold', pad=50)
    ax.set_xticks(x_pos_reason)
    ax.set_xticklabels(reasoning_groups, fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('plot7_reasoning_complexity.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def print_detailed_statistics(all_metrics, human_results, openai_results, anthropic_results, deepseek_results, human_metrics):
    """
    Print statistics shown in graphs
    
    Args:
        all_metrics: List of metric dictionaries for each model
        human_results: Results dictionary for human expert
        openai_results: Results dictionary for GPT-4o
        anthropic_results: Results dictionary for Claude Sonnet 4.5
        deepseek_results: Results dictionary for DeepSeek
        human_metrics: Metrics dictionary for human expert (for categories)
    """
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    for metrics in all_metrics:
        print(f"{metrics['model']}:")
        print(f"  Accuracy: {metrics['overall_accuracy']*100:.2f}%")
        
        results_list = [human_results, openai_results, anthropic_results, deepseek_results]
        model_idx = ['Human Expert', 'GPT-4o', 'Claude Sonnet 4.5', 'DeepSeek'].index(metrics['model'])
        result_dict = results_list[model_idx]
        
        all_ranks = []
        for category_data in result_dict['category_results'].values():
            for position in category_data['positions']:
                all_ranks.append(position['rank_distance'])
        mean_rank = sum(all_ranks) / len(all_ranks) if all_ranks else 0
        print(f"  Mean Rank Distance: {mean_rank:.2f}")
    
    print("\n" + "="*80)
    print("PER CATEGORY ACCURACY")
    print("="*80)
    
    categories = human_metrics['categories']
    for category in categories:
        print(f"\n{category}:")
        for metrics in all_metrics:
            cat_idx = metrics['categories'].index(category)
            acc = metrics['category_accuracies'][cat_idx] * 100
            print(f"  {metrics['model']}: {acc:.2f}%")
    
    print("\n" + "="*80)
    print("PER CATEGORY MEAN RANK DISTANCE")
    print("="*80)
    
    for category in categories:
        print(f"\n{category}:")
        for metrics in all_metrics:
            cat_idx = metrics['categories'].index(category)
            avg_rank = metrics['avg_rank_distances'][cat_idx]
            print(f"  {metrics['model']}: {avg_rank:.2f}")
    
    print("\n" + "="*80)
    print("REASONING COMPLEXITY")
    print("="*80)
    
    immediate_categories = ['P1 Win in 1', 'P2 Win in 1 (Block)']
    short_categories = ['Late Game', 'P1 Win in 3', 'P2 Win in 3']
    long_categories = ['P1 Win in 5', 'P2 Win in 5', 'Early Game']
    
    def get_avg_performance(categories_list):
        perf = []
        for metrics in all_metrics:
            cat_dict = dict(zip(metrics['categories'], metrics['category_accuracies']))
            accs = [cat_dict[cat] * 100 for cat in categories_list if cat in cat_dict]
            perf.append(sum(accs) / len(accs) if accs else 0)
        return perf
    
    immediate_perf = get_avg_performance(immediate_categories)
    short_perf = get_avg_performance(short_categories)
    long_perf = get_avg_performance(long_categories)
    
    print("\nImmediate Action:")
    for i, metrics in enumerate(all_metrics):
        print(f"  {metrics['model']}: {immediate_perf[i]:.2f}%")
    
    print("\nShort Planning:")
    for i, metrics in enumerate(all_metrics):
        print(f"  {metrics['model']}: {short_perf[i]:.2f}%")
    
    print("\nLong Planning:")
    for i, metrics in enumerate(all_metrics):
        print(f"  {metrics['model']}: {long_perf[i]:.2f}%")
    
    print("\n" + "="*80)

