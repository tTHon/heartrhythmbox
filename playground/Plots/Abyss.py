import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # --- 1. DATA PREPARATION ---
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

    # Reverse order for plotting top-to-bottom
    endpoints = endpoints[::-1]
    effect_sizes = effect_sizes[::-1]
    confidence_intervals = confidence_intervals[::-1]

# --- 2. COLOR THEME (Stitch & Sun Palette) ---
    # Since the background is bright (Sky/Sand), we need DARK text.
    
    TEXT_COLOR = '#0F2043'    # Deep Navy (Matches Stitch's outlines/eyes)
    BAR_COLOR = '#2E0245'     
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

        # C. The Effect Size Marker (Raincoat Yellow with Navy Outline)
        # A square with high contrast
        ax.scatter(effect, y_positions[i], 
                   c=MARKER_FILL, edgecolors=TEXT_COLOR, linewidth=2, 
                   s=450, zorder=3, marker='o')
        
        # D. The Data Label
        # Aligned to the right of the bar
        label_text = f'{effect:.1f} [{low:.1f}, {high:.1f}]'
        ax.text(6.2, y_positions[i], label_text, 
                va='center', ha='left', fontsize=32, color=TEXT_COLOR, 
                fontfamily='Inter') # Monospace aligns numbers perfectly

    # --- 4. AXES & STRUCTURE ---
    
    # Vertical "No Effect" line
    ax.axvline(x=0, color=TEXT_COLOR, linestyle='--', alpha=0.4, linewidth=1.5, zorder=1)

    # Y-Axis Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(endpoints, fontsize=32, fontweight='medium')
    ax.tick_params(axis='y', length=0, pad=15) # Remove ticks, push text slightly left
    
    # X-Axis Labels
    ax.set_xlabel('Risk Difference (95% CI)', fontsize=32, labelpad=15, fontweight='light')
    ax.tick_params(axis='x', length=10, pad=10, width=3) 
    
    # Clean Spines (Borders)
    # Remove everything except the bottom line
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['bottom'].set_color(TEXT_COLOR)

    # Set Layout limits
    ax.set_xlim(-2, 8) # Extended right side to make room for text labels
    
    plt.tight_layout()
    
    # Save command to demonstrate transparency works
    plt.savefig('playground/Plots/Abyss.png', dpi=300, transparent=True)
    
    #plt.show()