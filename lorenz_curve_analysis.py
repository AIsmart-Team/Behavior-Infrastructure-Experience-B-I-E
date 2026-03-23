import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import warnings

warnings.filterwarnings('ignore')


def calculate_fcmr_metrics(df, group_col):
    df['is_fast'] = pd.to_numeric(df['is_fast_charge_event'], errors='coerce').fillna(0)
    demand = df.groupby(group_col).agg({
        'is_fast': 'sum',
        group_col: 'count'
    }).rename(columns={'is_fast': 'delta_s', group_col: 'Delta_s'}).reset_index()

    if group_col == 'street':
        unique_stations = df.drop_duplicates('station_id').copy()
    else:
        unique_stations = df.drop_duplicates(group_col).copy()

    unique_stations['fast_pile_count'] = pd.to_numeric(unique_stations['fast_pile_count'], errors='coerce').fillna(0)
    unique_stations['slow_pile_count'] = pd.to_numeric(unique_stations['slow_pile_count'], errors='coerce').fillna(0)

    supply = unique_stations.groupby(group_col).agg({
        'fast_pile_count': 'sum',
        'slow_pile_count': 'sum'
    }).reset_index()
    supply['sigma_s'] = supply['fast_pile_count']
    supply['Sigma_s'] = supply['fast_pile_count'] + supply['slow_pile_count']

    merged = pd.merge(demand, supply[[group_col, 'sigma_s', 'Sigma_s']], on=group_col, how='inner')
    merged = merged[(merged['Delta_s'] > 0) & (merged['Sigma_s'] > 0)].copy()

    merged['ratio_demand'] = merged['delta_s'] / merged['Delta_s']
    merged['ratio_supply'] = merged['sigma_s'] / merged['Sigma_s']
    merged['m_i'] = (merged['ratio_demand'] - merged['ratio_supply']).abs()

    m_values = merged['m_i'].values
    n = len(m_values)
    m_sorted = np.sort(m_values)

    index = np.arange(1, n + 1)
    fc_gini = (2 * np.sum(index * m_sorted) / (n * np.sum(m_sorted) + 1e-6)) - (n + 1) / n

    lx = np.insert(np.arange(1, n + 1) / n, 0, 0)
    ly = np.insert(m_sorted.cumsum() / (m_sorted.sum() + 1e-6), 0, 0)

    return lx, ly, fc_gini


def main():
    CSV_PATH = 'data/Raw_data_all_new_select_clustered_gmm_classification_all.csv'
    OUTPUT_DIR = 'output/'
    OUTPUT_DATA_FILE = os.path.join(OUTPUT_DIR, 'Fig_Lorenz.csv')

    COLOR_STREET = '#0492e8'
    COLOR_STATION = '#d56d26'
    COLOR_EQUALITY = '#FF6B6B'

    LEGEND_X = -0.01
    LEGEND_Y = 0.99

    FIG_SIZE = (10, 9)
    DPI = 300
    FONT_SIZE = 20
    ZONE_FONT_SIZE = 18

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Data...")
    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    if 'station_id' not in df.columns:
        df['station_unique_str'] = df['app_name'].astype(str) + "_" + df['station_name'].astype(str)
        df['station_id'] = pd.factorize(df['station_unique_str'])[0]

    df = df.dropna(subset=['street', 'station_id']).copy()

    print("Calculating metrics...")
    lx_street, ly_street, gini_street = calculate_fcmr_metrics(df, 'street')
    lx_station, ly_station, gini_station = calculate_fcmr_metrics(df, 'station_id')

    print(f"Saving source data to {OUTPUT_DATA_FILE}...")
    df_street = pd.DataFrame({
        'Street_Cumulative_Share_Units': lx_street,
        'Street_Cumulative_Share_Mismatch': ly_street
    })
    df_station = pd.DataFrame({
        'Station_Cumulative_Share_Units': lx_station,
        'Station_Cumulative_Share_Mismatch': ly_station
    })

    df_fig = pd.concat([df_street, df_station], axis=1)
    df_fig.to_csv(OUTPUT_DATA_FILE, index=False, encoding='utf-8-sig')

    print("Plotting Lorenz Curve...")
    plt.rcParams['font.sans-serif'] = ['Arial']
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    rect_low = Rectangle((0, 0), 0.5, 0.25,
                         facecolor='#22850c', alpha=0.1, edgecolor='none', zorder=0)
    ax.add_patch(rect_low)
    ax.text(0.02, 0.23, 'Low Inequality',
            ha='left', va='top', fontsize=ZONE_FONT_SIZE,
            color='#22850c', alpha=0.99)

    rect_high = Rectangle((0.7, 0.7), 0.3, 0.3,
                          facecolor='#8a0a93', alpha=0.1, edgecolor='none', zorder=0)
    ax.add_patch(rect_high)
    ax.text(0.7, 0.98, 'High Inequality',
            ha='left', va='top', fontsize=ZONE_FONT_SIZE,
            color='#8a0a93', alpha=0.99)

    ax.plot([0, 1], [0, 1], color=COLOR_EQUALITY, linestyle='--',
            linewidth=2.5, label='Perfect Equality', zorder=2)

    ax.plot(lx_street, ly_street, color=COLOR_STREET, linewidth=3.5,
            label=f'Street-level (Gini={gini_street:.3f})', zorder=4)
    ax.fill_between(lx_street, lx_street, ly_street,
                    color=COLOR_STREET, alpha=0.15, zorder=3)

    ax.plot(lx_station, ly_station, color=COLOR_STATION, linewidth=3.5,
            label=f'Station-level (Gini={gini_station:.3f})', zorder=6)
    ax.fill_between(lx_station, lx_station, ly_station,
                    color=COLOR_STATION, alpha=0.08, zorder=5)

    ax.set_xlabel("Cumulative Share of Units", fontsize=FONT_SIZE)
    ax.set_ylabel("Cumulative Share of Mismatch Intensity", fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=FONT_SIZE, width=1.5, length=6)

    ax.legend(loc='upper left', bbox_to_anchor=(LEGEND_X, LEGEND_Y),
              fontsize=FONT_SIZE, frameon=True, edgecolor='grey')

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('#34495E')

    output_png = os.path.join(OUTPUT_DIR, "01_fcmr_lorenz_curve_enhanced.png")
    plt.savefig(output_png, bbox_inches='tight', facecolor='white', dpi=DPI)
    print(f"PNG saved: {output_png}")

    output_pdf = os.path.join(OUTPUT_DIR, "01_fcmr_lorenz_curve_enhanced.pdf")
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white', format='pdf')
    print(f"PDF saved: {output_pdf}")

    plt.close()

    print(f"\nAnalysis Completed.")
    print(f"Street FC-Gini: {gini_street:.4f}")
    print(f"Station FC-Gini: {gini_station:.4f}")


if __name__ == "__main__":
    main()