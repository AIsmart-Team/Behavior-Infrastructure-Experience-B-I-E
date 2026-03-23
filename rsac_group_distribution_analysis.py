import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
from scipy import stats as scipy_stats
from scipy import stats as scipy_stats_kde

warnings.filterwarnings('ignore')


def main():
    CSV_PATH = 'data/Raw_data_all_new_select_clustered_gmm_classification_all.csv'
    OUTPUT_DIR = 'rsac_group_analysis'
    OUTPUT_DATA_FILE_DIST = os.path.join(OUTPUT_DIR, 'Fig_RSAC_Group_Distributions_Data.csv')
    OUTPUT_DATA_FILE_HIST = os.path.join(OUTPUT_DIR, 'Fig_RSAC_Histogram_Data.csv')
    OUTPUT_FORMATS = ['png', 'pdf']

    ALPHA = 2.5
    BETA = 0.5
    EPSILON = 1.0

    DPI = 300
    FONT_NAME = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [FONT_NAME]
    plt.rcParams['axes.unicode_minus'] = False

    FS_LABEL = 22
    FS_TICK = 22
    FS_TEXT = 22

    PALETTE = ['#bcc6c9', '#add8e6', '#2b8cbe']
    CLASS_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

    HIST_BAR_COLOR = '#2b8cbe'
    HIST_BAR_EDGE_COLOR = '#ffffff'
    HIST_BAR_EDGE_WIDTH = 1.2
    HIST_KDE_COLOR = '#ff9246'
    HIST_KDE_WIDTH = 2.0
    HIST_VLINE_RISK_COLOR = '#d7191c'
    HIST_VLINE_PRIV_COLOR = '#1a9641'
    HIST_VLINE_WIDTH = 2

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

    df['wait_time'] = pd.to_numeric(df['wait_time'], errors='coerce').fillna(0)

    if 'evdata_vehicle_id' in df.columns:
        driver_col = 'evdata_vehicle_id'
    elif 'vehicle_id' in df.columns:
        driver_col = 'vehicle_id'
    else:
        df['vehicle_id'] = np.random.randint(0, 1000, len(df))
        driver_col = 'vehicle_id'

    print("Step 2: Calculating RSAC Index...")

    eta_city = df['wait_time'].mean()

    def get_mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else 0

    driver_stats = df.groupby(driver_col).agg({
        'R_time': 'mean',
        'R_space_norm': 'mean',
        'R_strategy': 'mean',
        'Time_Class': get_mode,
        'Space_Class': get_mode,
        'Strategy_Class': get_mode,
        'wait_time': 'mean',
        'station_id': 'count'
    }).rename(columns={'wait_time': 'eta_j', 'station_id': 'total_charges'})

    driver_stats['r_tau'] = driver_stats['R_time']

    if driver_stats['R_space_norm'].max() > 1.1:
        driver_stats['r_sigma'] = driver_stats['R_space_norm'] / 2.0
    else:
        driver_stats['r_sigma'] = driver_stats['R_space_norm']

    if driver_stats['R_strategy'].max() > 1.1:
        driver_stats['r_pi'] = driver_stats['R_strategy'] / 3.0
    else:
        driver_stats['r_pi'] = driver_stats['R_strategy']

    driver_stats['r_bar'] = (driver_stats['r_tau'] + driver_stats['r_sigma'] + driver_stats['r_pi']) / 3.0
    driver_stats['r_norm'] = np.sqrt(
        driver_stats['r_tau'] ** 2 + driver_stats['r_sigma'] ** 2 + driver_stats['r_pi'] ** 2)

    term1 = np.power(driver_stats['r_bar'], ALPHA)
    term2 = np.power((driver_stats['r_bar'] * np.sqrt(3)) / (driver_stats['r_norm'] + 1e-9), BETA)
    term3 = (eta_city + EPSILON) / (driver_stats['eta_j'] + EPSILON)

    driver_stats['Theta_j'] = term1 * term2 * term3

    cols_to_save = ['Theta_j', 'Time_Class', 'Space_Class', 'Strategy_Class', 'eta_j', 'total_charges']
    driver_stats[cols_to_save].to_csv(OUTPUT_DATA_FILE_DIST, index=True, encoding='utf-8-sig')
    print(f"Source data saved to: {OUTPUT_DATA_FILE_DIST}")

    def plot_regularity_distribution(data_df, class_col, title_dim):
        print(f"Plotting distribution for {title_dim}...")

        fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)

        plot_data = data_df.copy()
        plot_data['Label'] = plot_data[class_col].map(CLASS_LABELS)

        order = ['Low', 'Medium', 'High']

        sns.violinplot(x='Label', y='Theta_j', data=plot_data, order=order,
                       palette=PALETTE, inner=None, ax=ax, alpha=0.3, linewidth=0)

        sns.boxplot(x='Label', y='Theta_j', data=plot_data, order=order,
                    palette=PALETTE, width=0.2, ax=ax,
                    boxprops={'zorder': 2}, showfliers=False)

        groups = [plot_data[plot_data['Label'] == l]['Theta_j'] for l in order if l in plot_data['Label'].values]
        if len(groups) > 1:
            stat, p_val = scipy_stats.kruskal(*groups)
            p_text = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
            stats_msg = f"Kruskal-Wallis Test:\nH = {stat:.1f}\np = {p_text}"

            ax.text(0.05, 0.95, stats_msg, transform=ax.transAxes,
                    fontsize=FS_TEXT, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgrey'))

        ax.set_xlabel(f"{title_dim} Regularity Level", fontsize=FS_LABEL)
        ax.set_ylabel("RSAC Index", fontsize=FS_LABEL)

        ax.axhline(0.9, color='#d7191c', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(ax.get_xlim()[1], 0.9, ' 0.9\n(Risk)',
                va='top', ha='left', fontsize=20, color='#d7191c', rotation=90)

        ax.axhline(1.1, color='#1a9641', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(ax.get_xlim()[1], 1.1, ' 1.1\n(Privileged)',
                va='bottom', ha='left', fontsize=20, color='#1a9641', rotation=90)

        ax.tick_params(axis='both', labelsize=FS_TICK, direction='out', length=6, width=1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        means = plot_data.groupby('Label')['Theta_j'].mean()

        for i, label in enumerate(order):
            if label in means:
                ax.text(i, means[label], f"mean={means[label]:.2f}",
                        ha='center', va='bottom', fontsize=FS_TEXT, color='black')

        fname = f"rsac_dist_{title_dim.lower()}"
        for fmt in OUTPUT_FORMATS:
            save_path = os.path.join(OUTPUT_DIR, f"{fname}.{fmt}")
            plt.savefig(save_path, bbox_inches='tight', dpi=DPI)
        plt.close()

    print("Step 3: Generating Group Distribution Plots...")

    if 'Time_Class' in driver_stats.columns:
        plot_regularity_distribution(driver_stats, 'Time_Class', 'Time')

    if 'Space_Class' in driver_stats.columns:
        plot_regularity_distribution(driver_stats, 'Space_Class', 'Spatial')

    if 'Strategy_Class' in driver_stats.columns:
        plot_regularity_distribution(driver_stats, 'Strategy_Class', 'Strategic')

    print("Step 4: Generating Overall Histogram...")

    theta_values = driver_stats['Theta_j'].dropna()
    theta_values.to_csv(OUTPUT_DATA_FILE_HIST, index=True, header=['Theta_j'], encoding='utf-8-sig')
    print(f"Source data saved to: {OUTPUT_DATA_FILE_HIST}")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)

    sns.histplot(
        theta_values,
        bins=30,
        kde=False,
        color=HIST_BAR_COLOR,
        edgecolor=HIST_BAR_EDGE_COLOR,
        linewidth=HIST_BAR_EDGE_WIDTH,
        ax=ax,
        alpha=0.7
    )

    kde = scipy_stats_kde.gaussian_kde(theta_values)
    x_range = np.linspace(theta_values.min(), theta_values.max(), 200)
    kde_values = kde(x_range)

    ax2 = ax.twinx()
    kde_line = ax2.plot(x_range, kde_values, color=HIST_KDE_COLOR, linewidth=HIST_KDE_WIDTH, label='Density (KDE)')
    ax2.set_ylabel('Density', fontsize=FS_LABEL)
    ax2.tick_params(labelsize=FS_TICK)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_linewidth(1.2)
    ax2.grid(False)

    ax.set_xlabel("RSAC Index", fontsize=FS_LABEL)
    ax.set_ylabel("Frequency", fontsize=FS_LABEL)

    line1 = ax.axvline(0.9, color=HIST_VLINE_RISK_COLOR, linestyle='--', linewidth=HIST_VLINE_WIDTH,
                       label='Risk Threshold (0.9)')
    line2 = ax.axvline(1.1, color=HIST_VLINE_PRIV_COLOR, linestyle='--', linewidth=HIST_VLINE_WIDTH,
                       label='Privilege Threshold (1.1)')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=22, frameon=False, loc='upper right')

    ax.tick_params(labelsize=FS_TICK)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['top'].set_linewidth(1.2)

    ax.tick_params(axis='x', top=False, bottom=True, direction='out', length=6, width=1.2)
    ax.tick_params(axis='y', left=True, right=False, direction='out', length=6, width=1.2)

    for fmt in OUTPUT_FORMATS:
        save_path = os.path.join(OUTPUT_DIR, f"rsac_overall_hist.{fmt}")
        plt.savefig(save_path, bbox_inches='tight', dpi=DPI)
    plt.close()

    print(f"\nAnalysis Completed. Results in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()