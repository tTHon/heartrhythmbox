import matplotlib.pyplot as plt
import numpy as np

def create_forest_plot(endpoints, effect_sizes, confidence_intervals, p_values):
    """
    Create a forest plot for comparing two arms across multiple endpoints.
    
    Parameters:
    -----------
    endpoints : list of str
        Names of the endpoints being compared
    effect_sizes : list of float
        Effect sizes (e.g., hazard ratios, odds ratios)
    confidence_intervals : list of tuples
        Lower and upper confidence intervals for each effect size
    p_values : list of float
        P-values for each comparison
    """

    # Set figure size
    plt.figure(figsize=(20, 10))
    
    # Calculate number of endpoints and positions
    num_endpoints = len(endpoints)
    y_positions = np.arange(num_endpoints)
    
    # Create the plot
    plt.scatter(effect_sizes, y_positions, s=100, c=['blue' if p < 0.05 else 'gray' for p in p_values], zorder=3)
    
    # Add confidence intervals
    for i, (lower, upper) in enumerate(confidence_intervals):
        plt.plot([lower, upper], [y_positions[i], y_positions[i]], color='black', zorder=2)
        
    # Add vertical line at no effect (1.0)
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    # Customize the plot
    plt.yticks(y_positions, endpoints)
    plt.xlabel('Effect Size (95% CI)')
    plt.title('Forest Plot: Treatment Effect Across Endpoints')
    
    # Add p-values on the right side
    for i, p in enumerate(p_values):
        plt.text(plt.xlim()[1], i, f'p={p:.3f}', va='center')
        
    # Add grid
    plt.grid(True, axis='x', alpha=0.3)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    return plt.gcf()

# Example usage
if __name__ == "__main__":
    # Sample data
    endpoints = [
        "Primary Endpoint",
        "Death",
        "MI",
        "Stroke",
        "CV Hospitalization"
    ]
    
    effect_sizes = [2.8, 0.1, 0.1, -0.1, 2.3]
    
    confidence_intervals = [
        (0.09, 5.5),
        (-1.2, 1.4),
        (-0.9, 1.1),
        (-0.7, 0.6),
        (-0.1, 4.8)
    ]
    
    # p_values = [0.24, 0.07, 0.33, 0.56, 0.04]

    # Reverse the order of endpoints, effect_sizes, and confidence_intervals
    endpoints = endpoints[::-1]
    effect_sizes = effect_sizes[::-1]
    confidence_intervals = confidence_intervals[::-1]
    
    # Create and display the plot
    #fig = create_forest_plot(endpoints, effect_sizes, confidence_intervals, p_values)
    #fig = create_forest_plot(endpoints, effect_sizes, confidence_intervals)
    #plt.show()

    # Set default font properties and background color
    plt.rcParams.update({
    'font.size': 18,
    'font.family': 'sans-serif',
    'axes.facecolor': 'none',  # Background color for axes
    'figure.facecolor': 'none',  # Background color for figure
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'legend.facecolor': 'white',
    'legend.edgecolor': 'white'
    })

    # Create and display the plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # Calculate number of endpoints and positions
    num_endpoints = len(endpoints)
    y_positions = np.arange(num_endpoints)* 0.5  # Reduce space between bars

    # Create the plot with bars for 95% CI
    bar_height = 0.2
    scatter_size = 1000
    for i, (effect_size, (lower, upper)) in enumerate(zip(effect_sizes, confidence_intervals)):
        ax.barh(y_positions[i], upper - lower, left=lower, height=bar_height, color='cyan', alpha=0.9, edgecolor='none', zorder=2)
        ax.scatter(effect_size, y_positions[i], color='steelblue', edgecolors='white', s=scatter_size, zorder=3, marker= 's')
        ax.text(upper + 0.1, y_positions[i], f'{effect_size:.1f} [{lower:.1f}, {upper:.1f}]', va='center', fontsize=24, color='white')

    # Add vertical line at no effect (1.0)
    ax.axvline(x=0, color='whitesmoke', linestyle='-', alpha=0.5, linewidth=4, zorder=1)

    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(endpoints, fontsize=28)
    ax.tick_params(axis='y', width=2)  # Adjust the linewidth of the tick labels
    ax.set_xlabel('Risk Differences (95% CI)', fontsize=28)
    #ax.set_title('Forest Plot of Effect Sizes with 95% Confidence Intervals', fontsize=22)

    # Customize plot border color
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Set x-axis scale
    ax.set_xlim(-2, 6)  # Adjust the range as needed

    plt.tight_layout()
    plt.show()