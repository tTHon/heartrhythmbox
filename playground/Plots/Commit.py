import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # --- 1. DATA PREPARATION ---
    endpoints = [
        "Death",
        "Reinfarction",
        "VF",
    ]
    
    effect_sizes = [0.99, 0.82, 0.83]
    
    confidence_intervals = [
        (0.92, 1.05),
        (0.72, 0.92),
        (0.75, 0.93),
    ]

    # Added P-values from the COMMIT trial
    p_values = [0.69, 0.001, 0.001]

    # Reverse order for plotting top-to-bottom
    endpoints = endpoints[::-1]
    effect_sizes = effect_sizes[::-1]
    confidence_intervals = confidence_intervals[::-1]
    p_values = p_values[::-1]

# --- 2. COLOR THEME (Stitch & Sun Palette) ---
    # Since the background is bright (Sky/Sand), we need DARK text.
    
    TEXT_COLOR = '#F0EFEB'    # Off-White (Dark enough for bright bg) 
    BAR_COLOR = '#DFC8E4'      # Light Purple (Stitch's fur color)
    MARKER_FILL = '#FFD700'   # Raincoat Yellow (Matches Stitch's raincoat)
    
    plt.rcParams.update({
        'font.size': 28,
        'font.family': 'inter',
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
    fig, ax = plt.subplots(figsize=(16, 10))

    # Grid Setup
    y_positions = np.arange(len(endpoints))
    
    # Add a subtle vertical grid to help read x-values
    ax.grid(axis='x', color=TEXT_COLOR, alpha=0.2, linestyle='--', linewidth=2)

    # Add Column Labels (Headers)
    # We place these slightly above the top-most data point (max(y_positions) + 0.5)
    header_y = max(y_positions) + 0.6
    ax.text(1.2, header_y, "Odds Ratio [95% CI]", fontweight='bold', ha='left', fontsize=30)
    ax.text(1.75, header_y, "P-Value", fontweight='bold', ha='left', fontsize=30)

    # Plot Data
    for i, (effect, (low, high)) in enumerate(zip(effect_sizes, confidence_intervals)):
        
        # A. The Confidence Interval Line
        # We use a simple horizontal line instead of a bar for a cleaner look
        ax.plot([low, high], [y_positions[i], y_positions[i]], 
                color=BAR_COLOR, alpha=1.0, linewidth=4.5, zorder=2)
        
        # B. Vertical "Caps" at the ends of the CI (optional, adds precision look)
        cap_height = 0.1
        ax.plot([low, low], [y_positions[i]-cap_height, y_positions[i]+cap_height], 
                color=BAR_COLOR, alpha=1.0, linewidth=4.5, zorder=2)
        ax.plot([high, high], [y_positions[i]-cap_height, y_positions[i]+cap_height], 
                color=BAR_COLOR, alpha=1.0, linewidth=4.5, zorder=2)

        # C. The Effect Size Marker 
        # A square with high contrast
        ax.scatter(effect, y_positions[i], 
                   c=MARKER_FILL, edgecolors=TEXT_COLOR, linewidth=2, 
                   s=450, zorder=3, marker='o')
        
        # D. The Data Label
        # Format P-value to show <0.001 if very small
        p_val = p_values[i]
        p_text = f"p={p_val}" if p_val >= 0.001 else "p<0.001"
        label_text_OR = f'{effect:.2f} [{low:.2f}, {high:.2f}]'
        label_text_p = f'{p_text}'
        
        ax.text(1.2, y_positions[i], label_text_OR, 
                va='center', ha='left', fontsize=28, color=TEXT_COLOR)
        ax.text(1.75, y_positions[i], label_text_p, 
                va='center', ha='left', fontsize=28, color=TEXT_COLOR)
    # --- 4. AXES & STRUCTURE ---
    
    # Vertical "No Effect" line
    ax.axvline(x=1, color=TEXT_COLOR, linestyle='--', alpha=0.8, linewidth=2.5, zorder=1)

    # Y-Axis Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(endpoints, fontsize=32, fontweight='medium')
    ax.tick_params(axis='y', length=0, pad=15) # Remove ticks, push text slightly left
    
    # X-Axis Labels
    ax.set_xlabel('Odd Ratio (95% CI)', fontsize=32, labelpad=15, fontweight='light')
    ax.tick_params(axis='x', length=10, pad=10, width=3) 
    
    # Clean Spines (Borders)
    # Remove everything except the bottom line
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['bottom'].set_color(TEXT_COLOR)

    # Set Layout limits
    ax.set_xlim(0.6, 1.5) # Extended right side to make room for text labels
    
    plt.tight_layout()
    
    # Save command to demonstrate transparency works
    plt.savefig('playground/Plots/Commit.png', dpi=300, transparent=True)
    
    #plt.show()