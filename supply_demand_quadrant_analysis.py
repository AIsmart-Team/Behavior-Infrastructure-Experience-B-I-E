import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import warnings

warnings.filterwarnings('ignore')

CSV_PATH = 'data/Raw_data_all_new_select_clustered_gmm_classification_all.csv'
OUTPUT_DIR = 'output/'
DPI = 600
FORMATS = ['png', 'pdf']

STYLE = {
    'font.family': 'Arial',
    'size_min': 100,
    'size_max': 3000,
    'color_map': 'PiYG_r',
}

plt.rcParams['font.family'] = STYLE['font.family']


def calculate_metrics_consistent(df, group_col):
    total_events_city = len(df)
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

    res = pd.merge(demand, supply[[group_col, 'sigma_s', 'Sigma_s']], on=group_col, how='inner')
    res = res[(res['Delta_s'] > 0) & (res['Sigma_s'] > 0)].copy()

    res['ratio_demand'] = res['delta_s'] / res['Delta_s']
    res['ratio_supply'] = res['sigma_s'] / res['Sigma_s']
    res['W_FPG'] = (res['ratio_demand'] - res['ratio_supply']) * (res['Delta_s'] / total_events_city)

    return res


def plot_quadrant_scatter(df, filename_base, level_name):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)

    df['abs_w_fpg'] = df['W_FPG'].abs()
    df_plot = df.sort_values('abs_w_fpg', ascending=True)

    x = df_plot['ratio_supply']
    y = df_plot['ratio_demand']
    w_fpg = df_plot['W_FPG']
    sizes = (df_plot['Delta_s'] / df['Delta_s'].max()) * STYLE['size_max'] + STYLE['size_min']

    v_min, v_max = w_fpg.min(), w_fpg.max()
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=v_min, vmax=v_max)

    sc = ax.scatter(x, y, s=sizes, c=w_fpg, cmap=STYLE['color_map'], norm=norm,
                    alpha=0.8, edgecolors='black', linewidths=0.6, zorder=3)

    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.2, zorder=1, alpha=0.6)

    ax.text(0.12, 0.68, "Demand > Supply", fontsize=21, ha='left', fontweight='bold', color='#b91574')
    ax.text(0.88, 0.22, "Supply > Demand", fontsize=21, ha='right', fontweight='bold', color='#62b626')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fast-Charging Supply Ratio", fontsize=21)
    ax.set_ylabel("Fast-Charging Demand Ratio", fontsize=21)
    ax.tick_params(axis='both', labelsize=20)

    ticks = [v_min, v_min / 2, 0, v_max / 2, v_max]
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, ticks=ticks)

    cbar.outline.set_visible(False)

    cbar.set_label(f"W-FPG ({level_name})", fontsize=21)

    cbar.ax.set_yticklabels([f"{t * 100:.1f}" for t in ticks])

    cbar.ax.text(0.5, 1.03, r'$\times 10^{-2}$', transform=cbar.ax.transAxes,
                 ha='center', va='bottom', fontsize=20)

    cbar.ax.tick_params(labelsize=20, direction='out', length=4, width=1, color='black')

    plt.tight_layout()
    for fmt in FORMATS:
        save_path = os.path.join(OUTPUT_DIR, f"{filename_base}.{fmt}")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    plt.close()


def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("Loading Data...")
    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")
        return

    if 'station_id' not in df.columns:
        df['station_id'] = pd.factorize(df['app_name'].astype(str) + df['station_name'].astype(str))[0]

    print("Processing Station Level...")
    df_station = calculate_metrics_consistent(df, 'station_id')

    station_csv_path = os.path.join(OUTPUT_DIR, "Fig_Quadrant_Station_Data.csv")
    df_station.to_csv(station_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved station level data to: {station_csv_path}")

    plot_quadrant_scatter(df_station, "04_supply_demand_quadrant_scatter_station", "Station-Level")

    print("Processing Street Level...")
    df_street = calculate_metrics_consistent(df, 'street')

    street_csv_path = os.path.join(OUTPUT_DIR, "Fig_Quadrant_Street_Data.csv")
    df_street.to_csv(street_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved street level data to: {street_csv_path}")

    plot_quadrant_scatter(df_street, "04_supply_demand_quadrant_scatter", "Street-Level")

    print("\nAnalysis Completed.")


if __name__ == "__main__":
    main()