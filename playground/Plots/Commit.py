import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

if __name__ == "__main__":
    # --- 1. DATA PREPARATION ---
    endpoints = [
        "Stroke or Systemic Embolism",
        "All-Cause Death",
        "Cardiovascular Death",
    ]
    
    effect_sizes = [0.81, 0.92, 0.89]
    
    confidence_intervals = [
        (0.74, 0.89),
        (0.87, 0.97),
        (0.83, 0.96),
    ]

    # Added P-values from the COMMIT trial
    #p_values = [0.69, 0.001, 0.001]

    # Reverse order for plotting top-to-bottom
    endpoints = endpoints[::-1]
    effect_sizes = effect_sizes[::-1]
    confidence_intervals = confidence_intervals[::-1]
    #p_values = p_values[::-1]

# --- 2. COLOR THEME (Stitch & Sun Palette) ---
        
    TEXT_COLOR = '#eeeeea'    # Off-White (Dark enough for bright bg) 
    BAR_COLOR = '#0357a0'      # Light Purple (Stitch's fur color)
    MARKER_FILL = '#FED154'   # minion yellow
    
    plt.rcParams.update({
        'font.size': 36,
        'font.family': 'inter, sans-serif',
        'font.weight': 'regular',
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'axes.edgecolor': TEXT_COLOR,
        'figure.facecolor': 'none', # Transparent Figure
        'axes.facecolor': 'none',   # Transparent Axis
        'savefig.facecolor': 'none' # Transparent Export
    })

    # --- 3. PLOTTING ---
    fig, ax = plt.subplots(figsize=(20, 12), dpi=300)

    # Grid Setup
    y_positions = np.arange(len(endpoints))
    
    # Add a subtle vertical grid to help read x-values
    ax.grid(axis='x', color=TEXT_COLOR, alpha=0.2, linestyle='--', linewidth=2)

    # Add Column Labels (Headers)
    # We place these slightly above the top-most data point (max(y_positions) + 0.5)
    header_y = max(y_positions) + 0.3
    ax.text(1.45, header_y, "Hazard Ratio \n[95% CI]", fontweight='bold', ha='center', fontsize=36)
    #ax.text(1.75, header_y, "P-Value", fontweight='bold', ha='left', fontsize=30)

    # Plot Data
    for i, (effect, (low, high)) in enumerate(zip(effect_sizes, confidence_intervals)):
        
        # A. The Confidence Interval Line
        # We use a simple horizontal line instead of a bar for a cleaner look
        ax.plot([low, high], [y_positions[i], y_positions[i]], 
                color=BAR_COLOR, alpha=1.0, linewidth=8, zorder=2)
        
        # B. Vertical "Caps" at the ends of the CI (optional, adds precision look)
        cap_height = 0.05
        ax.plot([low, low], [y_positions[i]-cap_height, y_positions[i]+cap_height], 
                color=BAR_COLOR, alpha=1.0, linewidth=8, zorder=2)
        ax.plot([high, high], [y_positions[i]-cap_height, y_positions[i]+cap_height], 
                color=BAR_COLOR, alpha=1.0, linewidth=8, zorder=2)

        # C. The Effect Size Marker 
        # A square with high contrast
        ax.scatter(effect, y_positions[i], 
                   c=MARKER_FILL, edgecolors=TEXT_COLOR, linewidth=0, 
                   s=750, zorder=3, marker='o')
        
        # D. The Data Label
        # Format P-value to show <0.001 if very small
        #p_val = p_values[i]
        #p_text = f"p={p_val}" if p_val >= 0.001 else "p<0.001"
        label_text_OR = f'{effect:.2f} [{low:.2f}, {high:.2f}]'
        #label_text_p = f'{p_text}'
        
        ax.text(1.2, y_positions[i], label_text_OR, 
                va='center', ha='left', fontsize=36, color=TEXT_COLOR)
        #ax.text(1.75, y_positions[i], label_text_p, 
        #        va='center', ha='left', fontsize=28, color=TEXT_COLOR)
    # --- 4. AXES & STRUCTURE ---
    
    # Vertical "No Effect" line
    ax.axvline(x=1, color=TEXT_COLOR, linestyle='--', alpha=0.8, linewidth=2.5, zorder=1)

    # Y-Axis Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(endpoints, fontsize=36, fontweight='medium')
    ax.tick_params(axis='y', length=0, pad=0) # Remove ticks, push text slightly left
    ax.set_ylim(-0.25, len(endpoints) - 0.5)
    
    # X-Axis Labels
    ax.set_xlabel('Hazard Ratios of DOACs vs. Warfarin', fontsize=36, labelpad=15, fontweight='light')
    # Define X-axis limits for the plot area (Log Scale logic)
    x_min, x_max = 0.6, 1.75
    ax.set_xlim(x_min, x_max)
    ax.set_xscale('log') # Use log scale as per the original medical chart
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.5, 0.7, 1.0, 1.5]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(['0.5', '0.7', '1.0', '1.5']))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis='x', which='major', length=10, width=2, color=TEXT_COLOR)

    
    # Clean Spines (Borders)
    # Remove everything except the bottom line
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['bottom'].set_color(TEXT_COLOR)

    
    plt.tight_layout()
    
    # Save command to demonstrate transparency works
    plt.savefig('playground/Plots/Commit.png', dpi=300, transparent=True)
    
    #plt.show()