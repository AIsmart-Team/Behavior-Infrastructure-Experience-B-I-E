import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import seaborn as sns
import os
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# ==========================================================
# 1. Data Loading & Processing Module
# ==========================================================

def load_and_process_data(csv_path, target_dates):
    """
    Loads raw charging data and calculates hourly mismatch (u_it) for each station.
    """
    print("Step 1: Loading and processing raw data...")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, low_memory=False)
    df['charge_start_time'] = pd.to_datetime(df['charge_start_time'])
    df['date_str'] = df['charge_start_time'].dt.strftime('%Y-%m-%d')
    df['hour'] = df['charge_start_time'].dt.hour

    # Filter by date if specified
    if target_dates:
        df = df[df['date_str'].isin(target_dates)]

    # Calculate Supply Ratio (R_supply)
    if 'fast_pile_count' not in df.columns:
        # Fallback if pile counts aren't in columns
        station_supply = df.groupby('station_id').size().reset_index(name='cnt')
        station_supply['fast'] = 10;
        station_supply['slow'] = 10
    else:
        cols = ['station_id', 'fast_pile_count', 'slow_pile_count']
        station_supply = df.groupby('station_id')[cols[1:]].first().reset_index()
        station_supply['fast'] = pd.to_numeric(station_supply['fast_pile_count'], errors='coerce').fillna(0)
        station_supply['slow'] = pd.to_numeric(station_supply['slow_pile_count'], errors='coerce').fillna(0)

    station_supply['Sigma'] = station_supply['fast'] + station_supply['slow']
    station_supply = station_supply[station_supply['Sigma'] > 0].copy()
    station_supply['ratio_supply'] = station_supply['fast'] / station_supply['Sigma']
    supply_dict = station_supply.set_index('station_id')['ratio_supply'].to_dict()

    # Calculate Hourly Demand and Mismatch
    hourly_df = df.groupby(['station_id', 'date_str', 'hour']).agg({
        'is_fast_charge_event': 'sum',
        'station_id': 'count'
    }).rename(columns={'is_fast_charge_event': 'delta', 'station_id': 'Delta'}).reset_index()

    metrics_list = []
    for _, row in hourly_df.iterrows():
        sid = row['station_id']
        if sid in supply_dict:
            r_supply = supply_dict[sid]
            r_demand = row['delta'] / row['Delta'] if row['Delta'] > 0 else 0
            metrics_list.append({
                'station_id': sid,
                'date_str': row['date_str'],
                'hour': row['hour'],
                'u_it': r_demand - r_supply,
                'Delta': row['Delta']
            })

    return pd.DataFrame(metrics_list)


def classify_and_get_stats(hourly_df, output_dir):
    """
    Calculates S_i and B_i, classifies stations into Q1-Q4, and saves the summary CSV.
    """
    print("Step 2: Calculating S_i, B_i and Classifying Regimes...")
    stats = hourly_df.groupby('station_id').agg({
        'u_it': ['mean', 'std'],
        'Delta': 'sum'
    })
    stats.columns = ['S_i', 'B_i', 'Total_Volume']
    stats = stats.reset_index()
    stats['B_i'] = stats['B_i'].fillna(0)

    # Define Thresholds
    threshold_B = stats['B_i'].mean()
    threshold_S = 0

    def get_regime(row):
        if row['S_i'] > threshold_S:
            return 'Q1' if row['B_i'] > threshold_B else 'Q4'
        else:
            return 'Q2' if row['B_i'] > threshold_B else 'Q3'

    stats['Regime'] = stats.apply(get_regime, axis=1)

    # Save processed statistics
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    csv_save_path = os.path.join(output_dir, 'station_details_sorted.csv')
    stats.sort_values(by=['Regime', 'B_i'], ascending=[True, True]).to_csv(csv_save_path, index=False)
    print(f"Saved station classification details to: {csv_save_path}")

    return stats, threshold_B


def select_representatives(stats, manual_ids):
    """
    Selects one representative station per regime (Q1-Q4) for time-series plotting.
    """
    representatives = {}
    for r in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = stats[stats['Regime'] == r]
        selected_id = None

        # Try manual selection first
        if manual_ids.get(r) is not None:
            target_id = manual_ids[r]
            if target_id in stats['station_id'].values:
                selected_id = target_id
            else:
                print(f"Warning: Manual ID {target_id} for {r} not found in data.")

        # Fallback to max volume if manual ID fails or isn't provided
        if selected_id is None:
            if not subset.empty:
                selected_id = subset.loc[subset['Total_Volume'].idxmax(), 'station_id']
            else:
                representatives[r] = None
                continue

        row = stats[stats['station_id'] == selected_id].iloc[0]
        representatives[r] = {
            'id': row['station_id'],
            'S_i': row['S_i'],
            'B_i': row['B_i']
        }
    return representatives


# ==========================================================
# 2. Visualization Module: Time-Series (Figure C)
# ==========================================================

def draw_time_series_layout(mode, hourly_df, representatives, threshold_B, config, output_dir):
    """
    Draws the time-series mismatch profile (4x1 or 2x2 layout).
    """
    print(f"Step 3a: Plotting Time-Series Figure C ({mode} Layout)...")
    dates = sorted(config['target_dates'])
    date_labels = config['date_labels']
    regime_order = ['Q1', 'Q2', 'Q3', 'Q4']

    if mode == '4x1':
        nrows, ncols = 4, 1
        figsize = (14, 12)
        sharex, sharey = True, False
        legend_bbox = config['legend_bbox_4x1']
        hspace = config['hspace_4x1']
        wspace = None
    else:  # 2x2
        nrows, ncols = 2, 2
        figsize = (16, 10)
        sharex, sharey = True, True
        legend_bbox = config['legend_bbox_2x2']
        hspace = config['hspace_2x2']
        wspace = config['wspace_2x2']

    plt.rcParams['font.family'] = config['font_family']
    plt.rcParams['font.size'] = config['font_size_base']

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=config['dpi'],
                             sharex=sharex, sharey=sharey)
    axes = axes.flatten()

    for idx, regime in enumerate(regime_order):
        ax = axes[idx]
        rep_info = representatives[regime]
        color = config['colors'][regime]

        ax.set_facecolor('white')
        ax.grid(False)
        # Vertical day separators
        for d_i in range(len(dates) + 1):
            ax.axvline(x=d_i * 24, color='lightgray', linewidth=0.7, linestyle='-')
        # Zero line
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.6)

        if rep_info is None:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            continue

        station_id = rep_info['id']
        S_i = rep_info['S_i']
        B_i_val = rep_info['B_i']

        # Prepare data for plotting
        station_data = hourly_df[hourly_df['station_id'] == station_id].copy()
        plot_x, plot_y = [], []
        for i, d in enumerate(dates):
            day_data = station_data[station_data['date_str'] == d].set_index('hour')
            full_day = day_data.reindex(range(24))
            offset = i * 24
            plot_x.extend([h + offset for h in range(24)])
            vals = full_day['u_it'].fillna(method='ffill').fillna(0).values
            plot_y.extend(vals)

        # Plot Structural Mean line
        ax.axhline(y=S_i, color=color, linestyle='--', linewidth=1.5, alpha=0.9)
        # Plot Observed Mismatch line
        ax.plot(plot_x, plot_y, color=color, linewidth=1.5, alpha=0.9)

        # Fill area (Behavioral Volatility)
        y_array = np.array(plot_y)
        mask = ~np.isnan(y_array)
        ax.fill_between(plot_x, y_array, S_i, where=mask, interpolate=True, color=color, alpha=0.15)

        # Label Box
        b_desc = "High" if B_i_val > threshold_B else "Low"
        label_str = (f"{regime} (Station ID: {int(station_id)})\n"
                     f"($\mathcal{{S}}_i$={S_i:.3f}, {b_desc} $\mathcal{{B}}_i$={B_i_val:.3f})")

        ax.text(0.01, 0.9, label_str, transform=ax.transAxes,
                fontsize=config['font_size_label'],
                color=color, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))

        ax.set_ylim(-1.1, 1.1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

        # Y Label
        if mode == '4x1':
            ax.set_ylabel(r"Mismatch ($u_i(t)$)", fontsize=config['font_size_label'])
        else:
            if idx % 2 == 0:
                ax.set_ylabel(r"Mismatch ($u_i(t)$)", fontsize=config['font_size_label'])

        # X Ticks (0 and 12 for each day)
        major_ticks = []
        major_labels = []
        for i in range(len(dates)):
            base = i * 24
            major_ticks.extend([base, base + 12])
            major_labels.extend(['0', '12'])

        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_labels, fontsize=config['font_size_tick'])

        # X Axis Label Logic
        show_xlabel = False
        if mode == '4x1' and idx == 3: show_xlabel = True
        if mode == '2x2' and idx >= 2: show_xlabel = True

        if show_xlabel:
            ax.set_xlabel("Hour of Day (Cycle)", fontsize=config['font_size_label'])
            ax.tick_params(labelbottom=True)
        else:
            ax.tick_params(labelbottom=False)

        # Top Axis (Date Labels)
        show_top_axis = False
        if mode == '4x1' and idx == 0: show_top_axis = True
        if mode == '2x2' and idx < 2: show_top_axis = True

        if show_top_axis:
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            day_centers = [i * 24 + 12 for i in range(len(dates))]
            ax_top.set_xticks(day_centers)
            ax_top.set_xticklabels(date_labels, fontsize=config['font_size_tick'])
            ax_top.tick_params(axis='x', which='both', length=0, pad=4)
            for spine in ax_top.spines.values(): spine.set_visible(False)

    if mode == '4x1':
        plt.subplots_adjust(hspace=hspace)
    else:
        plt.subplots_adjust(hspace=hspace, wspace=wspace)

    # Common Legend
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.5, label=r'Observed $u_i(t)$'),
        Line2D([0], [0], color='black', lw=1.5, ls='--', label=r'Structural Mean $\mathcal{S}_i$'),
        Patch(facecolor='black', alpha=0.15, label='Behavioral Volatility')
    ]

    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=legend_bbox, ncol=3,
               fontsize=config['font_size_legend'], frameon=False)

    plt.subplots_adjust(top=0.90)

    suffix = "4x1" if mode == '4x1' else "2x2"
    save_png = os.path.join(output_dir, f"Figure_C_Typical_{suffix}.png")
    save_pdf = os.path.join(output_dir, f"Figure_C_Typical_{suffix}.pdf")

    plt.savefig(save_png, bbox_inches='tight')
    plt.savefig(save_pdf, bbox_inches='tight')
    print(f"Saved Time Series ({mode}): {save_png}")
    plt.close()


# ==========================================================
# 3. Visualization Module: Statistics (Violin & CDF)
# ==========================================================

def plot_violin(df, params, output_dir):
    """
    Plots independent violin charts for S_i and B_i.
    """
    print("Step 3b: Plotting Violin Charts...")
    plt.rcParams['font.family'] = params['font_family']
    plt.rcParams['font.size'] = params['font_size_base']

    metrics = [
        ('S_i', params['violin']['ylabel_left'], 'Si'),
        ('B_i', params['violin']['ylabel_right'], 'Bi')
    ]

    for metric_col, ylabel, suffix in metrics:
        fig, ax = plt.subplots(1, 1, figsize=params['violin']['figsize_single'], dpi=params['dpi'])

        sns.violinplot(
            data=df, x='Regime', y=metric_col, ax=ax,
            palette=params['colors'],
            order=params['violin']['regime_order'],
            scale=params['violin']['scale'],
            width=params['violin']['width'],
            inner=params['violin']['inner'],
            linewidth=params['violin']['linewidth'],
            saturation=params['violin']['saturation']
        )

        ax.set_xlabel('Regime', fontsize=params['font_size_base'])
        ax.set_ylabel(ylabel, fontsize=params['font_size_base'])

        if metric_col == 'S_i':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)

        ax.tick_params(labelsize=params['font_size_tick'])

        save_png = os.path.join(output_dir, f"violin_plot_{suffix}.png")
        save_pdf = os.path.join(output_dir, f"violin_plot_{suffix}.pdf")

        plt.tight_layout()
        plt.savefig(save_png, bbox_inches='tight')
        plt.savefig(save_pdf, bbox_inches='tight')
        print(f"Saved Violin ({suffix}): {save_png}")
        plt.close()


def plot_cdf(df, params, output_dir):
    """
    Plots independent CDF charts for S_i and B_i.
    """
    print("Step 3c: Plotting CDF Charts...")
    plt.rcParams['font.family'] = params['font_family']
    plt.rcParams['font.size'] = params['font_size_base']

    metrics = [
        ('S_i', params['cdf']['xlabel_left'], 'Si', params['cdf']['legend_bbox_left']),
        ('B_i', params['cdf']['xlabel_right'], 'Bi', params['cdf']['legend_bbox_right'])
    ]

    for metric_col, xlabel, suffix, legend_pos in metrics:
        fig, ax = plt.subplots(1, 1, figsize=params['cdf']['figsize_single'], dpi=params['dpi'])

        for regime in params['cdf']['regime_order']:
            subset = df[df['Regime'] == regime][metric_col].sort_values()
            if subset.empty: continue
            ax.plot(
                subset, np.linspace(0, 1, len(subset)),
                label=regime,
                linewidth=params['cdf']['line_width'],
                color=params['colors'][regime],
                alpha=params['cdf']['line_alpha']
            )

        ax.set_xlabel(xlabel, fontsize=params['font_size_base'])
        ax.set_ylabel('Cumulative Probability', fontsize=params['font_size_base'])
        ax.tick_params(labelsize=params['font_size_tick'])

        ax.legend(fontsize=params['font_size_legend'], frameon=False,
                  loc='upper left', bbox_to_anchor=legend_pos)

        save_png = os.path.join(output_dir, f"cdf_plot_{suffix}.png")
        save_pdf = os.path.join(output_dir, f"cdf_plot_{suffix}.pdf")

        plt.tight_layout()
        plt.savefig(save_png, bbox_inches='tight')
        plt.savefig(save_pdf, bbox_inches='tight')
        print(f"Saved CDF ({suffix}): {save_png}")
        plt.close()


# ==========================================================
# 4. Main Configuration & Execution
# ==========================================================

def main():
    # --- Paths ---
    csv_path = 'data/Raw_data_all_new_select_clustered_gmm_classification_all.csv'
    output_dir = 'Figure_C_Combined_Analysis'

    # --- Configuration Dictionary ---
    CONFIG = {
        'target_dates': [
            '2023-09-15', '2023-09-16', '2023-09-17',
            '2023-09-18', '2023-09-29', '2023-09-30'
        ],
        'date_labels': [
            'Day-1\n(Weekday)', 'Day-2\n(Weekend)', 'Day-3\n(Weekend)',
            'Day-4\n(Weekday)', 'Day-5\n(Holiday)', 'Day-6\n(Holiday)'
        ],
        'dpi': 300,
        'font_family': 'Arial',
        'font_size_base': 20,
        'font_size_label': 20,
        'font_size_tick': 18,
        'font_size_legend': 18,

        # Colors
        'colors': {
            'Q1': '#d62728', 'Q2': '#ff7f0e',
            'Q3': '#2ca02c', 'Q4': '#9467bd'
        },

        # Representative Selection
        'MANUAL_STATIONS': {
            'Q1': 10446, 'Q2': 11931, 'Q3': 12033, 'Q4': 11212
        },

        # Time Series Plot Settings
        'hspace_4x1': 0.1, 'hspace_2x2': 0.15, 'wspace_2x2': 0.15,
        'legend_bbox_4x1': (0.5, 1), 'legend_bbox_2x2': (0.5, 0.98),

        # Violin Settings
        'violin': {
            'figsize_single': (4.8, 4.5),
            'ylabel_left': r'$\mathcal{S}_i$ (Structural Mismatch)',
            'ylabel_right': r'$\mathcal{B}_i$ (Behavioral Mismatch)',
            'regime_order': ['Q1', 'Q2', 'Q3', 'Q4'],
            'scale': 'width', 'width': 0.8, 'inner': 'box',
            'linewidth': 1, 'saturation': 0.75
        },

        # CDF Settings
        'cdf': {
            'figsize_single': (4.8, 4.5),
            'xlabel_left': r'$\mathcal{S}_i$ (Structural Mismatch)',
            'xlabel_right': r'$\mathcal{B}_i$ (Behavioral Mismatch)',
            'regime_order': ['Q1', 'Q2', 'Q3', 'Q4'],
            'line_width': 2.5, 'line_alpha': 0.8,
            'legend_bbox_left': (0.005, 1), 'legend_bbox_right': (0.6, 0.55)
        }
    }

    # --- Execution Pipeline ---

    # 1. Load Data
    hourly_df = load_and_process_data(csv_path, CONFIG['target_dates'])

    if hourly_df.empty:
        print("Pipeline aborted due to data loading failure.")
        return

    # 2. Statistics & Classification
    stats_df, threshold_B = classify_and_get_stats(hourly_df, output_dir)

    # 3. Time Series Plots (Figure C)
    representatives = select_representatives(stats_df, CONFIG['MANUAL_STATIONS'])
    draw_time_series_layout('2x2', hourly_df, representatives, threshold_B, CONFIG, output_dir)
    draw_time_series_layout('4x1', hourly_df, representatives, threshold_B, CONFIG, output_dir)

    # 4. Statistical Plots (Violin & CDF)
    # Using the aggregated stats_df for distribution plotting
    plot_violin(stats_df, CONFIG, output_dir)
    plot_cdf(stats_df, CONFIG, output_dir)

    print("\n========================================================")
    print(f"All outputs successfully generated in: {output_dir}")
    print("========================================================")


if __name__ == "__main__":
    main()