import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def create_hfrref_forest_plot():
    # --- 1. DATA ENTRY ---
    # Labels (Treatment Arms)
    treatments = [
    "ARNI + ßB + MRA + SGLT2i",
    "ARNI + ßB + MRA",
    "ACEI + ßB + MRA + SGLT2i",
    "ACEI + ßB + Dig + H-ISDN",
    "ARNI + ßB + SGLT2i",
    "ACEI + ßB + MRA",
    "ARB + ßB + MRA",
    "ARNI + ßB",
    "ACEI + ARB + ßB + Dig",
    "ACEI + MRA + Dig",
    "ACEI + ßB + Dig",
    "ACEI + ßB",
    "ARB + ßB + Dig",
    "ARB + ßB",
    "ßB",
    ]

    # Hazard Ratios
    hrs = [0.39,0.45, 0.46, 0.47,0.52, 0.52,
      0.56, 0.59, 0.65, 0.66, 0.68, 0.69, 0.73, 0.74, 0.78]
    
    # Confidence Intervals (Lower, Upper)
    cis = [(0.32, 0.49),(0.37, 0.54), 
       (0.38, 0.56), (0.36, 0.61), (0.43, 0.63), 
       (0.44, 0.62), (0.47, 0.65), (0.51, 0.69), (0.55, 0.76), 
       (0.56, 0.78), (0.59, 0.78), (0.62, 0.77), (0.63, 0.83), 
       (0.66, 0.82), (0.72, 0.84) ]
    
    # Reverse order to plot top-to-bottom
    treatments = treatments[::-1]
    hrs = hrs[::-1]
    cis = cis[::-1]

    # --- 2. SETUP FIGURE & COLORS ---
    BG_COLOR = '#3C2847'  # The requested Dark Purple
    TEXT_COLOR = "#FBF4F4" # White for text
    MARKER_COLOR = '#F8DC00' # Bright Yellow for markers
    LINE_COLOR = '#FBF4F4' # White for lines
    
    plt.rcParams.update({
        'font.family': 'Roboto',
        'font.size': 32,
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': BG_COLOR,
        'savefig.facecolor': BG_COLOR
    })

    fig, ax = plt.subplots(figsize=(15, 15))

    # --- 3. PLOTTING ---
    y_pos = np.arange(len(treatments))
    
    # Define X-axis limits for the plot area (Log Scale logic)
    x_min, x_max = 0.1, 1.5
    ax.set_xlim(x_min, x_max)
    ax.set_xscale('log') # Use log scale as per the original medical chart
    
    # Custom ticks to match the image exactly
    ax.set_xticks([0.1, 0.5, 1.0, 1.5])
    ax.set_xticklabels(['0.1', '0.5', '1', '1.5'], fontsize=28, fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Loop to draw each line and marker
    for i, (hr, (low, high)) in enumerate(zip(hrs, cis)):
        
        # A. Draw the line (Confidence Interval)
        # Left side logic
        if low < x_min:
            ax.annotate('', xy=(x_min, y_pos[i]), xytext=(min(high, x_max), y_pos[i]),
                        arrowprops=dict(arrowstyle="<-", color=LINE_COLOR, lw=2.5))
        else:
            ax.plot([low, min(high, x_max)], [y_pos[i], y_pos[i]], color=LINE_COLOR, lw=2.5)
            ax.plot([low, low], [y_pos[i]-0.15, y_pos[i]+0.15], color=LINE_COLOR, lw=2.5)

        # Right side logic
        if high > x_max:
            ax.annotate('', xy=(x_max, y_pos[i]), xytext=(max(low, x_min), y_pos[i]),
                        arrowprops=dict(arrowstyle="->", color=LINE_COLOR, lw=2.5))
        else:
            if low >= x_min: 
                 ax.plot([low, high], [y_pos[i], y_pos[i]], color=LINE_COLOR, lw=2.5)
            ax.plot([high, high], [y_pos[i]-0.15, y_pos[i]+0.15], color=LINE_COLOR, lw=2.5)
        # B. Draw the Square Marker (Hazard Ratio)
        marker_size = 400
        ax.scatter(hr, y_pos[i], marker='s', s=marker_size, c=MARKER_COLOR, edgecolors='#5a005a', zorder=3)

        # C. Add Text Columns (HR and CI) on the right side
        text_str = f" {hr:.2f}  [{low:.2f} - {high:.2f}]"
        ax.text(1.75, y_pos[i], text_str, va='center', ha='left', fontsize=28, 
                color=TEXT_COLOR, fontfamily='Roboto')

    # --- 4. FORMATTING ---
    
    # Y-Axis Labels (Treatments)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(treatments, fontsize=30)
    
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
    ax.tick_params(axis='x', which='major', bottom=True, top=False, color=TEXT_COLOR, length=15, width=2.5)
    ax.tick_params(axis='y', length=0) # No y-ticks

    # Reference Line at 1.0
    ax.axvline(1.0, color=TEXT_COLOR, linestyle='-', linewidth=1, zorder=1)

    # Headers
    # "Treatment" (Top Left) aligned with the new left-aligned labels
    # We move the x-coordinate to roughly -0.55 (relative to axes) to match the pad=320
    ax.text(-0.3, 1.05, "Treatment", transform=ax.transAxes, 
            fontsize=32, fontweight='bold', ha='left')
    
    # "All-cause mortality" (Top Center-ish)
    ax.text(0.5, 1.05, "All-cause mortality", transform=ax.transAxes, 
            fontsize=32, fontweight='bold', ha='center')
            
    # "HR 95%-CI" (Top Right)
    ax.text(1.3, 1.05, "HR   95%-CI", transform=ax.transAxes, 
            fontsize=32, fontweight='bold', ha='center')
            
    # Layout adjustments to make room for the right-side text and left-side labels
    plt.subplots_adjust(right=0.75, left=0.35) 
    
    #plt.show()
    # Save command to demonstrate transparency works
    plt.savefig('playground/Plots/BBHFrEF.png', dpi=300, transparent=True)

if __name__ == "__main__":
    create_hfrref_forest_plot()