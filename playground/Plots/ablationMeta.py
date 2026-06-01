import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def create_af_ablation_stroke_plot():
    # --- 1. DATA ENTRY (EXTRACTED FROM THE ATTACHED FOREST PLOT) ---
    studies = [
        "CAMTAF 2014",
        "RAAFT-2 2014",
        "Sohara 2016",
        "CAMERA-MRI 2017",
        "CASTLE-AF 2018",
        "CABANA 2019",
        "CAPTAF 2019",
        "ATTEST 2020",
        "STOP AF First 2020",
        "CAPA 2021",
        "Cryo-FIRST 2021",
        "EARLY-AF 2021",
        "AVATAR 2022",
        "RAFT-AF 2022",
        "CASTLE-HTx 2023",
        "ORBITA-AF 2024",
        "Cho 2024",
        "SHAM-PVI 2024",
        "Total (Wald)"
    ]

    # Relative Risks (RR) - Using None for "Not estimable" entries
    rrs = [
        2.78, None, 2.18, None, 0.47, 0.68, 0.64, 0.33, None, 
        0.56, None, 0.19, None, 0.92, 0.33, None, 0.20, 0.32, 0.63
    ]
    
    # 95% Confidence Intervals (Lower, Upper)
    cis = [
        (0.12, 65.08), (None, None), (0.11, 44.44), (None, None), (0.17, 1.32),
        (0.42, 1.11), (0.11, 3.73), (0.01, 8.04), (None, None), (0.28, 1.12),
        (None, None), (0.01, 4.00), (None, None), (0.27, 3.13), (0.01, 8.08),
        (None, None), (0.01, 4.02), (0.01, 7.78), (0.45, 0.87)
    ]
    
    # Reverse order to plot top-to-bottom sequentially in Matplotlib
    studies = studies[::-1]
    rrs = rrs[::-1]
    cis = cis[::-1]

    # --- 2. SETUP FIGURE & COLORS ---
    BG_COLOR = '#3C2847'      # Dark Purple Theme
    TEXT_COLOR = "#FBF4F4"   # Clean Off-white
    MARKER_COLOR = '#F8DC00' # Bright Yellow for individual trials
    SUMMARY_COLOR = '#00E5FF'# Cyan Highlight for pooled summary diamond
    LINE_COLOR = '#FBF4F4'   
    
    plt.rcParams.update({
        'font.family': 'inter',
        'font.size': 30,
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': BG_COLOR,
        'savefig.facecolor': BG_COLOR
    })

    fig, ax = plt.subplots(figsize=(28, 22))

    # --- 3. PLOTTING LAYER ---
    y_pos = np.arange(len(studies))
    
    # Matches the exact 0.01 to 100 log-scale spectrum seen in the image
    x_min, x_max = 0.01, 100.0
    ax.set_xlim(x_min, x_max)
    ax.set_xscale('log')
    
    ax.set_xticks([0.01, 0.1, 1.0, 10.0, 100.0])
    ax.set_xticklabels(['0.01', '0.1', '1', '10', '100'], fontsize=26, fontweight='bold')

    # Draw markers, lines, and text annotations
    for i, (rr, ci) in enumerate(zip(rrs, cis)):
        is_total = "Total" in studies[i]
        
        if rr is not None:
            low, high = ci
            plot_low = max(low, x_min)
            plot_high = min(high, x_max)
            
            if not is_total:
                # Left bound cap/arrow
                if low < x_min:
                    ax.annotate('', xy=(x_min, y_pos[i]), xytext=(plot_high, y_pos[i]),
                                arrowprops=dict(arrowstyle="<-", color=LINE_COLOR, lw=2.5))
                else:
                    ax.plot([plot_low, plot_high], [y_pos[i], y_pos[i]], color=LINE_COLOR, lw=2.5)
                    ax.plot([plot_low, plot_low], [y_pos[i]-0.15, y_pos[i]+0.15], color=LINE_COLOR, lw=2.5)

                # Right bound cap/arrow
                if high > x_max:
                    ax.annotate('', xy=(x_max, y_pos[i]), xytext=(plot_low, y_pos[i]),
                                arrowprops=dict(arrowstyle="->", color=LINE_COLOR, lw=2.5))
                else:
                    if low >= x_min:
                        ax.plot([plot_low, plot_high], [y_pos[i], y_pos[i]], color=LINE_COLOR, lw=2.5)
                    ax.plot([plot_high, plot_high], [y_pos[i]-0.15, y_pos[i]+0.15], color=LINE_COLOR, lw=2.5)
                
                # Point Estimate Square
                ax.scatter(rr, y_pos[i], marker='o', s=500, c=MARKER_COLOR, zorder=3)
            else:
                # Summary Diamond Polygon coordinates
                diamond_x = [low, rr, high, rr]
                diamond_y = [y_pos[i], y_pos[i]+0.25, y_pos[i], y_pos[i]-0.25]
                ax.fill(diamond_x, diamond_y, color=SUMMARY_COLOR, edgecolor=SUMMARY_COLOR, lw=10, zorder=4)

            text_str = f"{rr:.2f} [{low:.2f}, {high:.2f}]"
        else:
            # Handles non-estimable study text safely
            text_str = "Not estimable"

        weight_style = 'bold' if is_total else 'normal'
        text_color = SUMMARY_COLOR if is_total else TEXT_COLOR
        textSize = 42 if is_total else 30
        ax.text(1.04, y_pos[i], text_str, va='center', ha='left', fontsize = textSize,
                color=text_color, fontweight=weight_style, transform=ax.get_yaxis_transform())

    # --- 4. FORMATTING & MARGIN DESIGN ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(studies, fontsize=26)
    
    # Left-align all row labels perfectly
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('left')
        if "Total" in label.get_text():
            label.set_weight('bold')
            label.set_color(SUMMARY_COLOR)
            label.set_fontsize(42)

    # Push study names outward clear of the plot framework bounding box
    ax.tick_params(axis='y', pad=420) 

    # Clean axes setup
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='x', which='major', bottom=True, top=False, color=TEXT_COLOR, length=12, width=2)
    ax.tick_params(axis='y', length=0)

    # Line of No Effect (Risk Ratio = 1.0)
    ax.axvline(1.0, color=TEXT_COLOR, linestyle='--', linewidth=1.5, zorder=1)

    # Section Table Headers
    ax.text(-0.50, 1.04, "Study or Subgroup", transform=ax.transAxes, fontweight='bold', ha='left')
    ax.text(0.49, 1.04, "Favors Catheter Ablation   |   Favors Medical Therapy", transform=ax.transAxes, fontweight='bold', ha='center')
    ax.text(1.21, 1.04, "Risk Ratio [95% CI]", transform=ax.transAxes, fontweight='bold', ha='center')
    
    # Graphic Header Title
    #ax.text(0.45, 1.09, "Forest Plot: Catheter Ablation vs Medical Therapy", transform=ax.transAxes, 
    #        fontsize=30, fontweight='bold', ha='center', color=SUMMARY_COLOR)

    # Dynamic adjustment block to optimize space layout dynamically
    plt.subplots_adjust(right=0.72, left=0.30, top=0.88, bottom=0.1) 
    
    # Save step outputs clean transparent render
    plt.savefig('playground/Plots/AF_Ablation_Stroke_Forest.png', dpi=300, transparent=True)

if __name__ == "__main__":
    create_af_ablation_stroke_plot()