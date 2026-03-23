import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import warnings

warnings.filterwarnings('ignore')


def load_or_generate_data(csv_path, target_dates):
    if os.path.exists(csv_path):
        print("Loading REAL data...")
        df = pd.read_csv(csv_path, low_memory=False)
        df['charge_start_time'] = pd.to_datetime(df['charge_start_time'])
        df['date_str'] = df['charge_start_time'].dt.strftime('%Y-%m-%d')
        if target_dates:
            df = df[df['date_str'].isin(target_dates)]

        if 'fast_pile_count' in df.columns:
            station_static = df.groupby('station_id').agg({
                'fast_pile_count': 'first',
                'slow_pile_count': 'first'
            }).reset_index()
        else:
            station_static = df.groupby('station_id').size().reset_index(name='count')
            station_static['fast_pile_count'] = 10
            station_static['slow_pile_count'] = 10

        station_static['Sigma'] = station_static['fast_pile_count'] + station_static['slow_pile_count']
        station_static = station_static[station_static['Sigma'] > 0].copy()
        station_static['R_supply'] = station_static['fast_pile_count'] / station_static['Sigma']

        df['hour'] = df['charge_start_time'].dt.hour
        hourly_stats = df.groupby(['station_id', 'date_str', 'hour']).agg({
            'is_fast_charge_event': 'sum',
            'station_id': 'count'
        }).rename(columns={'is_fast_charge_event': 'delta', 'station_id': 'Delta'}).reset_index()

        hourly_stats = hourly_stats.merge(station_static[['station_id', 'R_supply']], on='station_id', how='left')

        hourly_stats['R_demand'] = hourly_stats['delta'] / hourly_stats['Delta']
        hourly_stats['u_it'] = hourly_stats['R_demand'] - hourly_stats['R_supply']

        n_days = df['date_str'].nunique()
        return hourly_stats, n_days

    else:
        print("Generating SYNTHETIC data for demo...")
        np.random.seed(42)
        n_stations = 200
        n_days = 7
        records = []
        for i in range(n_stations):
            s_i = np.random.normal(0, 0.3)
            b_i = np.random.gamma(2, 0.05)
            daily_vol = int(np.random.exponential(50)) + 10
            records.append({
                'station_id': 1000 + i,
                'S_i': s_i,
                'B_i': b_i,
                'Total_Volume': daily_vol * n_days
            })
        return pd.DataFrame(records), n_days


def calculate_metrics(hourly_df, n_days, cfg, use_synthetic=False):
    if use_synthetic:
        metrics = hourly_df.copy()
        if 'Daily_Events' not in metrics.columns:
            metrics['Daily_Events'] = metrics['Total_Volume'] / n_days
    else:
        metrics = hourly_df.groupby('station_id').agg({
            'u_it': ['mean', 'std'],
            'Delta': 'sum'
        })
        metrics.columns = ['S_i', 'B_i', 'Total_Volume']
        metrics = metrics.reset_index()
        metrics['B_i'] = metrics['B_i'].fillna(0)

    min_volume = cfg.get('min_volume_threshold', 5)
    print(f"Filtering out stations with Total Volume < {min_volume}...")
    metrics = metrics[metrics['Total_Volume'] >= min_volume].copy()

    metrics['Daily_Events'] = metrics['Total_Volume'] / n_days

    return metrics


def plot_figure_A(df, cfg, output_dir):
    print("Plotting Figure A...")

    x_thresh = 0
    y_thresh = 0.05
    print(f"Thresholds -> S_i(X): {x_thresh}, B_i(Y-Fixed): {y_thresh}")

    def get_regime(row):
        if row['S_i'] > x_thresh:
            return 'Q1' if row['B_i'] > y_thresh else 'Q4'
        else:
            return 'Q2' if row['B_i'] > y_thresh else 'Q3'

    df['Regime'] = df.apply(get_regime, axis=1)

    total_stations = len(df)
    counts = df['Regime'].value_counts()
    percentages = (counts / total_stations * 100).round(1)

    plt.rcParams['font.family'] = cfg['font_family']
    plt.rcParams['font.size'] = cfg['font_size_base']
    fig, ax = plt.subplots(figsize=cfg['figsize'], dpi=cfg['dpi'])

    ax.axvline(x=x_thresh, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.axhline(y=y_thresh, color='black', linestyle='--', linewidth=1.2, alpha=0.6)

    size_min, size_max = cfg['bubble_size_range']
    vol_min, vol_max = df['Daily_Events'].min(), df['Daily_Events'].max()

    def map_size(val):
        if vol_max == vol_min: return size_min
        return size_min + (val - vol_min) / (vol_max - vol_min) * (size_max - size_min)

    df['plot_size'] = df['Daily_Events'].apply(map_size)

    regime_order = ['Q1', 'Q2', 'Q3', 'Q4']
    for regime in regime_order:
        subset = df[df['Regime'] == regime]
        if subset.empty: continue
        color = cfg['colors'][regime]
        ax.scatter(subset['S_i'], subset['B_i'],
                   s=subset['plot_size'],
                   c=color,
                   alpha=cfg['bubble_alpha'],
                   edgecolors='white',
                   linewidth=0.5)

    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        pos = cfg['text_pos'][q]
        ha = 'right' if pos[0] > 0.5 else 'left'
        va = 'top' if pos[1] > 0.5 else 'bottom'

        ax.text(pos[0], pos[1], cfg['quadrant_texts'][q],
                transform=ax.transAxes, ha=ha, va=va,
                color='black', fontsize=cfg['font_size_text'])

    ax.set_xlabel(r'Structural Mismatch ($\mathcal{S}_i$)', fontsize=cfg['font_size_label'])
    ax.set_ylabel(r'Behavioral Mismatch ($\mathcal{B}_i$)', fontsize=cfg['font_size_label'])

    ax.tick_params(axis='both', labelsize=18)

    ax.set_ylim(-0.2, 0.7)

    ref_vals = [int(vol_min), int((vol_min + vol_max) / 2), int(vol_max)]
    ref_sizes = [map_size(v) for v in ref_vals]

    size_handles = [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='gray', markersize=np.sqrt(s),
                           label=str(v), alpha=0.6) for v, s in zip(ref_vals, ref_sizes)]

    leg1 = ax.legend(handles=size_handles,
                     bbox_to_anchor=cfg['bubble_legend_bbox'],
                     loc='upper left',
                     title="Daily Events",
                     fontsize=18,
                     title_fontsize=18,
                     frameon=False, framealpha=0.8, edgecolor='none')

    ratio_handles = []
    for regime in regime_order:
        pct = percentages.get(regime, 0.0)
        color = cfg['colors'][regime]
        label_text = f"{regime}: {pct}%"
        h = Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                   markersize=10, label=label_text)
        ratio_handles.append(h)

    ax.add_artist(leg1)

    leg2 = ax.legend(handles=ratio_handles,
                     bbox_to_anchor=cfg['ratio_legend_bbox'],
                     loc='upper left',
                     title="Proportion",
                     fontsize=18,
                     title_fontsize=18,
                     frameon=False,
                     handletextpad=0.1,
                     labelspacing=0.8)

    plt.tight_layout()

    png_path = os.path.join(output_dir, 'Figure_A_Mechanism_Matrix_Fixed.png')
    plt.savefig(png_path, bbox_inches='tight')
    print(f"Figure A (PNG) saved to: {png_path}")

    pdf_path = os.path.join(output_dir, 'Figure_A_Mechanism_Matrix_Fixed.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()


def main():
    csv_path = 'data/Raw_data_all_new_select_clustered_gmm_classification_all.csv'
    output_dir = 'output/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    target_dates = [
        '2023-09-15', '2023-09-16', '2023-09-17', '2023-09-18',
        '2023-09-19', '2023-09-29', '2023-09-30'
    ]

    CONFIG = {
        'figsize': (12, 6),
        'dpi': 600,
        'font_family': 'Arial',
        'font_size_base': 20,
        'font_size_label': 20,
        'font_size_text': 20,

        'min_volume_threshold': 5,

        'colors': {
            'Q1': '#d62728',
            'Q2': '#ff7f0e',
            'Q3': '#9467bd',
            'Q4': '#2ca02c'
        },

        'bubble_size_range': (20, 600),
        'bubble_alpha': 0.75,

        'bubble_legend_bbox': (1.02, 1.0),
        'ratio_legend_bbox': (0.98, 0.6),

        'quadrant_texts': {
            'Q1': r'Q1: Volatile Shortage' + '\n' + r'($\mathcal{S}_i$>0, High $\mathcal{B}_i$)',
            'Q2': r'Q2: Volatile Surplus' + '\n' + r'($\mathcal{S}_i$<0, High $\mathcal{B}_i$)',
            'Q3': r'Q3: Stable Surplus' + '\n' + r'($\mathcal{S}_i$<0, Low $\mathcal{B}_i$)',
            'Q4': r'Q4: Stable Shortage' + '\n' + r'($\mathcal{S}_i$>0, Low $\mathcal{B}_i$)'
        },

        'text_pos': {
            'Q1': (0.98, 0.98),
            'Q2': (0.02, 0.98),
            'Q3': (0.02, 0.02),
            'Q4': (0.98, 0.02)
        }
    }

    data, n_days = load_or_generate_data(csv_path, target_dates)
    is_synthetic = not os.path.exists(csv_path)

    metrics = calculate_metrics(data, n_days, CONFIG, use_synthetic=is_synthetic)

    output_data_path = os.path.join(output_dir, 'Fig_Structural_Behavioral_Data.csv')
    metrics.to_csv(output_data_path, index=False, encoding='utf-8-sig')
    print(f"Source plotting data saved to: {output_data_path}")

    plot_figure_A(metrics, CONFIG, output_dir)


if __name__ == "__main__":
    main()