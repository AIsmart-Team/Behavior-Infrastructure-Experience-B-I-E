import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


def main():
    CSV_PATH = 'data/Raw_data_all_new_select_clustered_gmm_classification_all.csv'
    OUTPUT_DIR = 'output/'
    OUTPUT_DATA_FILE = os.path.join(OUTPUT_DIR, 'Fig.csv')
    OUTPUT_FORMATS = ['png', 'pdf']

    EPSILON = 1e-6

    DPI = 300
    FONT_NAME = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [FONT_NAME]

    FONT_SIZE_LABEL = 20
    FONT_SIZE_TICK = 20
    FONT_SIZE_LEGEND = 20

    AXES_LINE_WIDTH = 1.2
    TICK_LENGTH = 5
    TICK_DIRECTION = 'out'

    SCATTER_FIG_SIZE = (8, 7)
    SCATTER_CMAP = 'PiYG_r'
    SCATTER_ALPHA = 1
    SCATTER_EDGE_ALPHA = 1.0
    SCATTER_SIZE_MIN = 30
    SCATTER_SIZE_MAX = 600
    SCATTER_EDGE_WIDTH = 0.8
    SCATTER_LABEL_FONTSIZE = 20
    SCATTER_TICK_FONTSIZE = 20
    SCATTER_LEGEND_FONTSIZE = 20
    SCATTER_TEXT_FONTSIZE = 14

    DIAGONAL_COLOR = 'black'
    DIAGONAL_STYLE = '--'
    DIAGONAL_WIDTH = 1.5

    TEXT_SHORTAGE = "Demand > Supply"
    TEXT_SURPLUS = "Supply > Demand"
    TEXT_POS_SHORTAGE = (0.3, 0.7)
    TEXT_POS_SURPLUS = (0.7, 0.3)

    CDF_FIG_SIZE = (10, 9)
    CDF_LINE_COLOR = '#2E86AB'
    CDF_LINE_WIDTH = 3.0

    CDF_THRESHOLD_HIGH = 0.7
    CDF_THRESHOLD_HIGH_COLOR = '#d62728'
    CDF_THRESHOLD_HIGH_STYLE = '--'
    CDF_HIGH_TEXT_POS = (0.07, 0.84)

    CDF_THRESHOLD_LOW = 0.2
    CDF_THRESHOLD_LOW_COLOR = '#2ca02c'
    CDF_THRESHOLD_LOW_STYLE = '--'
    CDF_LOW_TEXT_POS = (0.22, 0.2)

    CDF_LABEL_FONTSIZE = 20
    CDF_TICK_FONTSIZE = 20
    CDF_TEXT_FONTSIZE = 20

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Step 1: Loading Data...")
    if not os.path.exists(CSV_PATH):
        alt_path = 'data/Raw_data_all_new_select_clustered_gmm_classification.csv'
        if os.path.exists(alt_path):
            CSV_PATH = alt_path

    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df = df.dropna(subset=['street', 'station_id']).copy()

    print("Step 2: Calculating FCMR Metrics...")

    df['is_fast'] = pd.to_numeric(df['is_fast_charge_event'], errors='coerce').fillna(0)
    demand_stats = df.groupby('street').agg({
        'is_fast': 'sum',
        'station_id': 'count'
    }).rename(columns={'is_fast': 'delta_s', 'station_id': 'Delta_s'}).reset_index()

    unique_stations = df.drop_duplicates('station_id')[
        ['street', 'station_id', 'fast_pile_count', 'slow_pile_count']].copy()
    unique_stations['fast_pile_count'] = pd.to_numeric(unique_stations['fast_pile_count'], errors='coerce').fillna(0)
    unique_stations['slow_pile_count'] = pd.to_numeric(unique_stations['slow_pile_count'], errors='coerce').fillna(0)

    supply_stats = unique_stations.groupby('street').agg({
        'fast_pile_count': 'sum',
        'slow_pile_count': 'sum'
    }).reset_index()
    supply_stats['sigma_s'] = supply_stats['fast_pile_count']
    supply_stats['Sigma_s'] = supply_stats['fast_pile_count'] + supply_stats['slow_pile_count']

    fcmr_df = pd.merge(demand_stats, supply_stats[['street', 'sigma_s', 'Sigma_s']], on='street', how='inner')
    fcmr_df = fcmr_df[(fcmr_df['Delta_s'] > 0) & (fcmr_df['Sigma_s'] > 0)].copy()

    fcmr_df['ratio_demand'] = fcmr_df['delta_s'] / fcmr_df['Delta_s']
    fcmr_df['ratio_supply'] = fcmr_df['sigma_s'] / fcmr_df['Sigma_s']
    fcmr_df['raw_gap'] = fcmr_df['ratio_demand'] - fcmr_df['ratio_supply']
    fcmr_df['m_i'] = fcmr_df['raw_gap'].abs()

    delta_total = fcmr_df['Delta_s'].values
    min_d = np.min(delta_total)
    max_d = np.max(delta_total)
    fcmr_df['weight_eta'] = (fcmr_df['Delta_s'] - min_d) / (max_d - min_d + EPSILON)
    fcmr_df['W_FPG'] = fcmr_df['raw_gap'] * fcmr_df['weight_eta']

    print(f"Saving plotting data to {OUTPUT_DATA_FILE}...")
    fcmr_df.to_csv(OUTPUT_DATA_FILE, index=False, encoding='utf-8-sig')

    print("Step 3: Plotting Supply-Demand Structural Quadrant Scatter Plot...")

    fig, ax = plt.subplots(figsize=SCATTER_FIG_SIZE, dpi=DPI)

    x = fcmr_df['ratio_supply']
    y = fcmr_df['ratio_demand']

    sizes = (fcmr_df['Delta_s'] / fcmr_df['Delta_s'].max()) * SCATTER_SIZE_MAX + SCATTER_SIZE_MIN

    v_limit_scatter = max(abs(fcmr_df['W_FPG'].min()), abs(fcmr_df['W_FPG'].max()))
    norm_scatter = plt.Normalize(vmin=-v_limit_scatter, vmax=v_limit_scatter)
    cmap_scatter = plt.get_cmap(SCATTER_CMAP)

    colors = cmap_scatter(norm_scatter(fcmr_df['W_FPG'].values))

    sc = ax.scatter(x, y, s=sizes, c=fcmr_df['W_FPG'], cmap=SCATTER_CMAP, norm=norm_scatter,
                    alpha=SCATTER_ALPHA, edgecolors='black', linewidths=1)

    for i, (xi, yi, si) in enumerate(zip(x, y, sizes)):
        edge_color = colors[i].copy()
        edge_color[3] = SCATTER_EDGE_ALPHA
        ax.scatter(xi, yi, s=si, facecolors='none', edgecolors=[edge_color],
                   linewidths=SCATTER_EDGE_WIDTH, zorder=3)

    ax.plot([0, 1], [0, 1], color=DIAGONAL_COLOR, linestyle=DIAGONAL_STYLE,
            linewidth=DIAGONAL_WIDTH, zorder=0)

    ax.text(TEXT_POS_SHORTAGE[0], TEXT_POS_SHORTAGE[1], TEXT_SHORTAGE,
            fontsize=18, color='black', ha='center', va='center')
    ax.text(TEXT_POS_SURPLUS[0], TEXT_POS_SURPLUS[1], TEXT_SURPLUS,
            fontsize=18, color='black', ha='center', va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel("Fast-Charging Supply Ratio", fontsize=SCATTER_LABEL_FONTSIZE)
    ax.set_ylabel("Fast-Charging Demand Ratio", fontsize=SCATTER_LABEL_FONTSIZE)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXES_LINE_WIDTH)
        spine.set_color('black')

    ax.tick_params(axis='both', direction=TICK_DIRECTION, length=TICK_LENGTH, width=AXES_LINE_WIDTH,
                   labelsize=SCATTER_TICK_FONTSIZE, top=False, right=False)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("W-FPG (Mismatch Intensity)",
                   fontsize=SCATTER_LEGEND_FONTSIZE)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=SCATTER_TICK_FONTSIZE)

    save_plot(plt, OUTPUT_DIR, "04_supply_demand_quadrant_scatter", OUTPUT_FORMATS)

    print("Step 4: Plotting FCMR CDF...")

    fig, ax = plt.subplots(figsize=CDF_FIG_SIZE, dpi=DPI)

    m_i_sorted = np.sort(fcmr_df['m_i'].values)
    cdf_values = np.arange(1, len(m_i_sorted) + 1) / len(m_i_sorted)

    sns.ecdfplot(data=fcmr_df, x='m_i', color=CDF_LINE_COLOR, linewidth=CDF_LINE_WIDTH, ax=ax)

    prop_high_mismatch = (fcmr_df['m_i'] > CDF_THRESHOLD_HIGH).mean()
    idx_high = np.searchsorted(m_i_sorted, CDF_THRESHOLD_HIGH, side='right') - 1
    if idx_high < 0:
        idx_high = 0
    cdf_at_high = cdf_values[idx_high] if idx_high < len(cdf_values) else 0

    prop_low_mismatch = (fcmr_df['m_i'] < CDF_THRESHOLD_LOW).mean()
    idx_low = np.searchsorted(m_i_sorted, CDF_THRESHOLD_LOW, side='right') - 1
    if idx_low < 0:
        idx_low = 0
    cdf_at_low = cdf_values[idx_low] if idx_low < len(cdf_values) else 0

    ax.plot([CDF_THRESHOLD_HIGH, CDF_THRESHOLD_HIGH], [0, cdf_at_high],
            color=CDF_THRESHOLD_HIGH_COLOR, linestyle=CDF_THRESHOLD_HIGH_STYLE,
            linewidth=2, label=f'High Threshold ({CDF_THRESHOLD_HIGH})')

    ax.plot([0, CDF_THRESHOLD_HIGH], [cdf_at_high, cdf_at_high],
            color=CDF_THRESHOLD_HIGH_COLOR, linestyle=CDF_THRESHOLD_HIGH_STYLE,
            linewidth=2)

    ax.text(CDF_HIGH_TEXT_POS[0], CDF_HIGH_TEXT_POS[1],
            f"High Mismatch Ratio (> {CDF_THRESHOLD_HIGH}): {prop_high_mismatch:.2%}",
            fontsize=CDF_TEXT_FONTSIZE, color=CDF_THRESHOLD_HIGH_COLOR,
            va='center', fontweight='bold')

    ax.plot([CDF_THRESHOLD_LOW, CDF_THRESHOLD_LOW], [0, cdf_at_low],
            color=CDF_THRESHOLD_LOW_COLOR, linestyle=CDF_THRESHOLD_LOW_STYLE,
            linewidth=2, label=f'Low Threshold ({CDF_THRESHOLD_LOW})')

    ax.plot([0, CDF_THRESHOLD_LOW], [cdf_at_low, cdf_at_low],
            color=CDF_THRESHOLD_LOW_COLOR, linestyle=CDF_THRESHOLD_LOW_STYLE,
            linewidth=2)

    ax.text(CDF_LOW_TEXT_POS[0], CDF_LOW_TEXT_POS[1],
            f"Low Mismatch Ratio (< {CDF_THRESHOLD_LOW}): {prop_low_mismatch:.2%}",
            fontsize=CDF_TEXT_FONTSIZE, color=CDF_THRESHOLD_LOW_COLOR,
            va='center', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Absolute Mismatch Intensity", fontsize=CDF_LABEL_FONTSIZE)
    ax.set_ylabel("Cumulative Proportion", fontsize=CDF_LABEL_FONTSIZE)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXES_LINE_WIDTH)
        spine.set_color('black')

    ax.tick_params(axis='both', direction=TICK_DIRECTION, length=TICK_LENGTH, width=AXES_LINE_WIDTH,
                   labelsize=CDF_TICK_FONTSIZE, top=False, right=False)

    save_plot(plt, OUTPUT_DIR, "05_fcmr_mismatch_cdf", OUTPUT_FORMATS)

    print(f"\nSuccess! All plots saved to: {os.path.abspath(OUTPUT_DIR)}")


def save_plot(plt_obj, output_dir, filename, formats):
    for fmt in formats:
        plt_obj.savefig(os.path.join(output_dir, f"{filename}.{fmt}"),
                        dpi=300, format=fmt, bbox_inches='tight')
    plt_obj.close()


if __name__ == "__main__":
    main()