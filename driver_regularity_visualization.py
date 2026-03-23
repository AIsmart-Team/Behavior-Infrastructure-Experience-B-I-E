import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

INPUT_DIR = 'source_data/'
OUTPUT_DIR_PLOTS = 'source_data/V2-source data Fig. 2a/'
FILE_FEATURES = os.path.join(INPUT_DIR, 'source data Fig. 2a.csv')

ABSOLUTE_THRESHOLDS = {
    'High': 0.75,
    'Medium': 0.5
}


def load_data(features_file):
    print(f"Loading features file: {features_file}")
    try:
        df_merged = pd.read_csv(features_file, index_col='evdata_vehicle_id')
        if 'Overall_Class' not in df_merged.columns:
            raise KeyError("Key 'Overall_Class' not found in input file.")
        print(f"Data loaded successfully. Total drivers: {len(df_merged)}")
        return df_merged
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def setup_plotting_styles(style_config):
    plt.style.use('seaborn-v0_8-white')

    plt.rcParams['font.family'] = style_config['font_family']
    plt.rcParams['font.size'] = style_config['tick_fontsize']
    plt.rcParams['text.color'] = style_config['font_color']
    plt.rcParams['axes.labelcolor'] = style_config['font_color']
    plt.rcParams['xtick.color'] = style_config['font_color']
    plt.rcParams['ytick.color'] = style_config['font_color']
    plt.rcParams['axes.titlecolor'] = style_config['font_color']

    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.color'] = 'none'
    plt.rcParams['grid.alpha'] = 0.0

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6


def save_plot_formats(fig, output_path_base):
    try:
        fig.savefig(f"{output_path_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{output_path_base}.pdf", bbox_inches='tight')
        fig.savefig(f"{output_path_base}.svg", bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Failed to save plot: {e}")


def plot_distributions(df_merged, thresholds, style_config, output_dir):
    print("Generating distribution plots...")

    metrics = [
        ('R_total', 'Overall_Class', 'Overall Dimension'),
        ('R_time_norm', 'Time_Class', 'Time Dimension'),
        ('R_space_norm', 'Space_Class', 'Space Dimension'),
        ('R_strategy', 'Strategy_Class', 'Strategy Dimension')
    ]

    th_high = thresholds['High']
    th_medium = thresholds['Medium']

    high_color = style_config['high_color']
    medium_color = style_config['medium_color']
    low_color = style_config['low_color']

    for metric, class_col, title in metrics:
        print(f"Processing: {title}")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        data_col = df_merged[metric]

        sns.histplot(data_col, kde=True, ax=ax, color='lightgray',
                     line_kws={'color': style_config['font_color'], 'linewidth': 0},
                     bins=40, alpha=0.1)

        for patch in ax.patches:
            center = patch.get_x() + patch.get_width() / 2

            if center >= th_high:
                patch.set_facecolor(high_color)
            elif center >= th_medium:
                patch.set_facecolor(medium_color)
            else:
                patch.set_facecolor(low_color)

            patch.set_edgecolor('white')
            patch.set_alpha(0.99)

        counts = df_merged[class_col].value_counts(normalize=True) * 100
        pct_low = counts.get('Low', 0)
        pct_medium = counts.get('Medium', 0)
        pct_high = counts.get('High', 0)

        handles = [
            mpatches.Patch(color=high_color, label=f'High ({pct_high:.1f}%)', alpha=0.9, edgecolor='black'),
            mpatches.Patch(color=medium_color, label=f'Medium ({pct_medium:.1f}%)', alpha=0.9, edgecolor='black'),
            mpatches.Patch(color=low_color, label=f'Low ({pct_low:.1f}%)', alpha=0.9, edgecolor='black')
        ]

        if metric in ['R_time_norm', 'R_total']:
            legend_loc = 'upper right'
        else:
            legend_loc = 'upper left'

        ax.legend(
            handles=handles,
            title=None,
            loc=legend_loc,
            fontsize=style_config['legend_fontsize']
        )

        ax.set_xlabel(f'Regularity Score ({title})',
                      fontsize=style_config['label_fontsize'],
                      color=style_config['font_color'])
        ax.set_ylabel('Density',
                      fontsize=style_config['label_fontsize'],
                      color=style_config['font_color'])

        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True)

        output_base_path = os.path.join(output_dir, f'V2-Simple_Plot_01_{metric}')
        save_plot_formats(fig, output_base_path)
        plt.close(fig)

    print("Distribution plots generated.")


def main():
    print("\n" + "=" * 60)
    print("Driver Regularity Visualization")
    print("=" * 60)

    STYLE_CONFIG = {
        'font_family': 'Arial',
        'font_color': 'black',
        'title_fontsize': 25,
        'label_fontsize': 25,
        'tick_fontsize': 25,
        'legend_fontsize': 23,

        'high_color': '#2b8cbe',
        'medium_color': '#add8e6',
        'low_color': '#d9d9d9'
    }

    os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR_PLOTS}")

    setup_plotting_styles(STYLE_CONFIG)

    df_merged = load_data(FILE_FEATURES)

    if df_merged is None:
        print("Analysis terminated due to data loading error.")
        return

    plot_distributions(
        df_merged,
        ABSOLUTE_THRESHOLDS,
        STYLE_CONFIG,
        OUTPUT_DIR_PLOTS
    )

    print("\n" + "=" * 60)
    print("Visualization Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()