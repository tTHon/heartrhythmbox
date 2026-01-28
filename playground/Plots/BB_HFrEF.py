import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def create_hfrref_forest_plot():
    # --- 1. DATA ENTRY ---
    # Labels (Treatment Arms)
    treatments = [
        "ARB + BB + MRA",
        "ARNI + BB + MRA + SGLT2 + Vericiguat",
        "ARNI + BB + MRA + SGLT2",
        "ARNI + BB + MRA + Vericiguat",
        "ARNI + BB + MRA",
        "ARNI + BB + MRA + Omecamtiv",
        "ACEI + BB + MRA + IVA",
        "ACEI + BB + MRA + SGLT2",
        "ACEI + BB + MRA + Vericiguat",
        "ARNI + BB + SGLT2",
        "ACEI + BB + MRA",
        "ACEI + BB + MRA + Omecamtiv",
        "ARNI + BB"
    ]

    # Hazard Ratios
    hrs = [0.42, 0.53, 0.60, 0.60, 0.68, 0.68, 0.70, 0.71, 0.71, 0.75, 0.80, 0.81, 0.85]

    # Confidence Intervals (Lower, Upper)
    cis = [
        (0.17, 1.04),
        (0.41, 0.69),
        (0.47, 0.77),
        (0.47, 0.76),
        (0.55, 0.85),
        (0.54, 0.86),
        (0.32, 1.51),
        (0.57, 0.88),
        (0.57, 0.88),
        (0.64, 0.87),
        (0.66, 0.97),
        (0.65, 0.99),
        (0.77, 0.94)
    ]

    # Reverse order to plot top-to-bottom
    treatments = treatments[::-1]
    hrs = hrs[::-1]
    cis = cis[::-1]

    # --- 2. SETUP FIGURE & COLORS ---
    BG_COLOR = '#3C2847'  # The requested Dark Purple
    TEXT_COLOR = '#FFFFFF' # White for contrast
    MARKER_COLOR = '#D1D5DB' # Light Gray/Silver for the squares
    LINE_COLOR = '#FFFFFF'
    
    plt.rcParams.update({
        'font.family': 'Roboto',
        'font.size': 16,
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': BG_COLOR,
        'savefig.facecolor': BG_COLOR
    })

    fig, ax = plt.subplots(figsize=(16, 9))

    # --- 3. PLOTTING ---
    y_pos = np.arange(len(treatments))
    
    # Define X-axis limits for the plot area (Log Scale logic)
    x_min, x_max = 0.2, 1.5
    ax.set_xlim(x_min, x_max)
    ax.set_xscale('log') # Use log scale as per the original medical chart
    
    # Custom ticks to match the image exactly
    ax.set_xticks([0.2, 0.5, 1.0, 1.5])
    ax.set_xticklabels(['0.2', '0.5', '1', '1.5'], fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Loop to draw each line and marker
    for i, (hr, (low, high)) in enumerate(zip(hrs, cis)):
        
        # A. Draw the line (Confidence Interval)
        # Left side logic
        if low < x_min:
            ax.annotate('', xy=(x_min, y_pos[i]), xytext=(min(high, x_max), y_pos[i]),
                        arrowprops=dict(arrowstyle="<-", color=LINE_COLOR, lw=1.5))
        else:
            ax.plot([low, min(high, x_max)], [y_pos[i], y_pos[i]], color=LINE_COLOR, lw=1.5)
            ax.plot([low, low], [y_pos[i]-0.15, y_pos[i]+0.15], color=LINE_COLOR, lw=1.5)

        # Right side logic
        if high > x_max:
            ax.annotate('', xy=(x_max, y_pos[i]), xytext=(max(low, x_min), y_pos[i]),
                        arrowprops=dict(arrowstyle="->", color=LINE_COLOR, lw=1.5))
        else:
            if low >= x_min: 
                 ax.plot([low, high], [y_pos[i], y_pos[i]], color=LINE_COLOR, lw=1.5)
            ax.plot([high, high], [y_pos[i]-0.15, y_pos[i]+0.15], color=LINE_COLOR, lw=1.5)

        # B. Draw the Square Marker (Hazard Ratio)
        marker_size = 120
        ax.scatter(hr, y_pos[i], marker='s', s=marker_size, c=MARKER_COLOR, edgecolors='none', zorder=3)

        # C. Add Text Columns (HR and CI) on the right side
        text_str = f"{hr:.2f} [{low:.2f}; {high:.2f}]"
        ax.text(1.6, y_pos[i], text_str, va='center', ha='left', fontsize=12, 
                color=TEXT_COLOR, fontfamily='monospace')

    # --- 4. FORMATTING ---
    
    # Y-Axis Labels (Treatments)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(treatments, fontsize=12)
    
    # ALIGN LEFT LOGIC:
    # 1. Get the current labels
    labels = ax.get_yticklabels()
    # 2. Iterate and set alignment
    for label in labels:
        label.set_horizontalalignment('left')
    # 3. Adjust padding to push them to the left edge of the margin area
    # Negative padding moves the text closer to the left edge of the figure
    ax.tick_params(axis='y', pad=320) 

    # Remove standard spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Customize bottom spine
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_linewidth(1)
    # Add small ticks on bottom
    ax.tick_params(axis='x', which='both', bottom=True, top=False, color=TEXT_COLOR)
    ax.tick_params(axis='y', length=0) # No y-ticks

    # Reference Line at 1.0
    ax.axvline(1.0, color=TEXT_COLOR, linestyle='-', linewidth=1, zorder=1)

    # Headers
    # "Treatment" (Top Left) aligned with the new left-aligned labels
    # We move the x-coordinate to roughly -0.55 (relative to axes) to match the pad=320
    ax.text(-0.55, 1.05, "Treatment", transform=ax.transAxes, 
            fontsize=14, fontweight='bold', ha='left')
    
    # "All-cause mortality" (Top Center-ish)
    ax.text(0.5, 1.05, "All-cause mortality", transform=ax.transAxes, 
            fontsize=14, fontweight='bold', ha='center')
            
    # "HR 95%-CI" (Top Right)
    ax.text(1.12, 1.05, "HR   95%-CI", transform=ax.transAxes, 
            fontsize=14, fontweight='bold', ha='center')

    # Layout adjustments to make room for the right-side text and left-side labels
    plt.subplots_adjust(right=0.75, left=0.35) 
    
    #plt.show()
    # Save command to demonstrate transparency works
    plt.savefig('playground/Plots/BBHFrEF.png', dpi=300, transparent=True)

if __name__ == "__main__":
    create_hfrref_forest_plot()