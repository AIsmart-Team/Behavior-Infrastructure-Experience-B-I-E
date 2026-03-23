import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import ConnectionPatch
import numpy as np
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')


def main():
    # ==========================================================================
    #                              1. 参数配置区域 (Configuration)
    # ==========================================================================

    # -------- A. 输入输出路径 --------
    CSV_PATH = 'data/Raw_data_all_new_select_clustered_gmm_classification_all.csv'
    OUTPUT_DIR = 'CACL_Selected_Plots_Data'
    OUTPUT_FORMATS = ['png', 'pdf']

    # -------- B. CACL 计算参数 --------
    W_FOOD_1KM, W_PUBLIC_1KM, W_AUTO_1KM, W_HOTEL_1KM = 0.30, 0.20, 0.30, 0.10
    W_FOOD_5KM, W_PUBLIC_5KM, W_AUTO_5KM, W_HOTEL_5KM = 0.025, 0.025, 0.025, 0.025
    SMOOTHING_K = 1
    ROBUST_QUANTILE = 0.99

    # -------- C. 视觉基础参数 --------
    DPI = 300
    FONT_NAME = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [FONT_NAME]
    plt.rcParams['axes.unicode_minus'] = False

    # -------- D. 散点图配置 (Scatter) --------
    QUAD_COLORS = {'Q1': '#2ca02c', 'Q2': '#1f77b4', 'Q3': '#7f7f7f', 'Q4': '#d62728'}
    QUAD_THRESH_X = 0.3
    QUAD_THRESH_Y = 0.3
    ISO_LEVELS = [0.3, 0.5, 0.7]
    ISO_COLORS = ['#d73027', '#fee08b', '#1a9850']
    ISO_WIDTHS = [2.0, 2.5, 3.0]
    GLOBAL_MEAN_POS = (0.60, 0.88)

    # -------- E. 堆积柱状图配置 --------
    BIN_EDGES = [0.0, 0.1, 0.2, 0.4, 1.01]
    BIN_LABELS = ['0.0-0.1', '0.1-0.2', '0.2-0.4', '0.4-1.0']

    # Time Load (Blues)
    STACK_COLORS_TIME = ['#eff3ff', '#6baed6', '#3182bd', '#08519c']
    # Amenity Score (Greens)
    STACK_COLORS_AMENITY = ['#edf8e9', '#74c476', '#31a354', '#006d2c']

    BAR_LEGEND_POS = (0.95, 0.95)

    # ==========================================================================
    #                              2. 数据加载与预处理
    # ==========================================================================

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Step 1: Loading Data...")
    if not os.path.exists(CSV_PATH):
        alt_path = 'data/Raw_data_all_new_select_clustered_gmm_classification.csv'
        if os.path.exists(alt_path): CSV_PATH = alt_path

    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # ID 类型转换
    if 'street_ID' in df.columns: df['street_ID'] = pd.to_numeric(df['street_ID'], errors='coerce')

    # Station ID 生成
    if 'station_id' not in df.columns:
        df['app_name'] = df['app_name'].fillna('Unknown')
        name_col = 'station_name' if 'station_name' in df.columns else df.columns[0]
        df['station_unique_str'] = df['app_name'].astype(str) + "_" + df[name_col].astype(str)
        df['station_id'] = pd.factorize(df['station_unique_str'])[0]

    # 时间计算
    df['wait_time'] = pd.to_numeric(df['wait_time'], errors='coerce').fillna(0)
    df['charge_duration'] = pd.to_numeric(df['charge_duration'], errors='coerce').fillna(0)
    df['total_time_cost'] = df['charge_duration'] + df['wait_time']

    poi_cols = ['food_km1', 'public_km1', 'auto_km1', 'hotel_km1',
                'food_km5', 'public_km5', 'auto_km5', 'hotel_km5']

    # ==========================================================================
    #                              3. CACL 计算核心
    # ==========================================================================

    def calculate_CACL(group_df, level_name):
        # 1. Demand D_i
        load = group_df.agg({'total_time_cost': 'sum', 'station_id': 'count'}).rename(
            columns={'total_time_cost': 'D_i', 'station_id': 'Event_Count'})

        d_min = load['D_i'].min()
        d_robust_max = load['D_i'].quantile(ROBUST_QUANTILE)
        if d_robust_max == d_min: d_robust_max = load['D_i'].max()
        load['D_norm'] = ((load['D_i'] - d_min) / (d_robust_max - d_min)).clip(0, 1)

        # 2. Amenity A_i
        if level_name == 'street':
            unique_sts = df.drop_duplicates('station_id')[['street_ID'] + poi_cols]
            poi_data = unique_sts.groupby('street_ID')[poi_cols].mean()
        else:
            poi_data = df.groupby('station_id')[poi_cols].first()

        def norm_robust(s, k=1):
            s_min = s.min()
            s_max = s.quantile(ROBUST_QUANTILE)
            if s_max <= s_min: s_max = s.max()
            return ((s + k - (s_min + k)) / ((s_max + k) - (s_min + k))).clip(0, 1)

        a_score = (
                norm_robust(poi_data['food_km1'], SMOOTHING_K) * W_FOOD_1KM +
                norm_robust(poi_data['public_km1'], SMOOTHING_K) * W_PUBLIC_1KM +
                norm_robust(poi_data['auto_km1'], SMOOTHING_K) * W_AUTO_1KM +
                norm_robust(poi_data['hotel_km1'], SMOOTHING_K) * W_HOTEL_1KM +
                norm_robust(poi_data['food_km5']) * W_FOOD_5KM +
                norm_robust(poi_data['public_km5']) * W_PUBLIC_5KM +
                norm_robust(poi_data['auto_km5']) * W_AUTO_5KM +
                norm_robust(poi_data['hotel_km5']) * W_HOTEL_5KM
        )

        # 3. Merge
        res = pd.merge(load, pd.DataFrame(a_score, columns=['A_i']), left_index=True, right_index=True)
        res['CACL'] = np.sqrt(res['D_norm'] * res['A_i'])

        # Quadrant
        def get_quad(row):
            if row['D_norm'] >= QUAD_THRESH_X and row['A_i'] >= QUAD_THRESH_Y: return 'Q1'
            if row['D_norm'] < QUAD_THRESH_X and row['A_i'] >= QUAD_THRESH_Y: return 'Q2'
            if row['D_norm'] < QUAD_THRESH_X and row['A_i'] < QUAD_THRESH_Y: return 'Q3'
            return 'Q4'

        res['Quad'] = res.apply(get_quad, axis=1)
        res['Color'] = res['Quad'].map(QUAD_COLORS)

        return res.reset_index()

    print("Step 2: Calculating Metrics...")
    df_street = calculate_CACL(df.groupby('street_ID'), 'street')
    df_station = calculate_CACL(df.groupby('station_id'), 'station')

    # -------- 输出源数据 (Source Data) --------
    station_csv_path = os.path.join(OUTPUT_DIR, 'Fig_CACL_Scatter_Station_Data.csv')
    street_csv_path = os.path.join(OUTPUT_DIR, 'Fig_CACL_Scatter_Street_Data.csv')

    df_station.to_csv(station_csv_path, index=False, encoding='utf-8-sig')
    df_street.to_csv(street_csv_path, index=False, encoding='utf-8-sig')
    print(f"Source data saved:\n  - {station_csv_path}\n  - {street_csv_path}")

    # ==========================================================================
    #                              4. 散点图绘制
    # ==========================================================================
    def plot_scatter(data_df, level_name):
        print(f"Plotting Scatter ({level_name})...")
        fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI)

        x, y = data_df['D_norm'], data_df['A_i']
        sizes = (data_df['Event_Count'] / data_df['Event_Count'].max()) * 400 + 50
        ax.scatter(x, y, s=sizes, c=data_df['Color'], alpha=0.7, edgecolors='white', linewidth=0.5)

        ax.axvline(QUAD_THRESH_X, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axhline(QUAD_THRESH_Y, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)

        x_line = np.linspace(0.001, 1.1, 500)
        for val, color, lw in zip(ISO_LEVELS, ISO_COLORS, ISO_WIDTHS):
            y_line = (val ** 2) / x_line
            valid = (y_line <= 1.1) & (x_line <= 1.1)
            ax.plot(x_line[valid], y_line[valid], color=color, linewidth=lw, linestyle='-')

            # 等值线标签
            label_x = 0.75
            label_y = (val ** 2) / label_x
            if label_y <= 1.0:
                ax.text(label_x + 0.02, label_y + 0.01, f'CACL={val}',
                        color=color,
                        fontsize=16,
                        ha='center',
                        va='bottom',
                        alpha=0.7,
                        rotation=-20,
                        bbox=dict(
                            boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor='none',
                            alpha=0.8
                        ))

        ax.text(0.98, 0.98, "Q1", transform=ax.transAxes, ha='right', va='top', fontsize=24, fontweight='bold',
                color=QUAD_COLORS['Q1'])
        ax.text(0.02, 0.98, "Q2", transform=ax.transAxes, ha='left', va='top', fontsize=24, fontweight='bold',
                color=QUAD_COLORS['Q2'])
        ax.text(0.02, 0.02, "Q3", transform=ax.transAxes, ha='left', va='bottom', fontsize=24, fontweight='bold',
                color=QUAD_COLORS['Q3'])
        ax.text(0.98, 0.02, "Q4", transform=ax.transAxes, ha='right', va='bottom', fontsize=24, fontweight='bold',
                color=QUAD_COLORS['Q4'])

        mean_val = data_df['CACL'].mean()
        ax.text(GLOBAL_MEAN_POS[0], GLOBAL_MEAN_POS[1], f"Mean CACL: {mean_val:.3f}",
                transform=ax.transAxes, fontsize=18,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

        ax.set_xlabel("Normalized Charging demand", fontsize=20)
        ax.set_ylabel("Service Level", fontsize=20)
        ax.tick_params(labelsize=18)
        ax.set_xlim(-0.02, 1.02);
        ax.set_ylim(-0.02, 1.02)
        ax.grid(False)

        filename = f"CACL_scatter_{level_name}"
        for fmt in OUTPUT_FORMATS:
            plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.{fmt}"), bbox_inches='tight', dpi=DPI)
        plt.close()

    plot_scatter(df_street, 'street')
    plot_scatter(df_station, 'station')

    # ==========================================================================
    #                              5. 堆积柱状图绘制
    # ==========================================================================
    def plot_stacked_distribution(data_df, level_name, stack_col, stack_name, colors_list):
        print(f"Plotting Stacked Bar: CACL vs {stack_name} ({level_name})...")

        # 1. Binning
        df_plot = data_df.copy()
        df_plot['CACL_Bin'] = pd.cut(df_plot['CACL'], bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True)
        df_plot['Stack_Bin'] = pd.cut(df_plot[stack_col], bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True)

        # 2. Crosstab (Count)
        ct = pd.crosstab(df_plot['CACL_Bin'], df_plot['Stack_Bin'])

        # 3. 创建主图和放大图的子图布局
        fig = plt.figure(figsize=(9, 7), dpi=DPI)

        ax_main = plt.subplot2grid((1, 10), (0, 0), colspan=7)
        ax_zoom = plt.subplot2grid((1, 10), (0, 7), colspan=3)

        # ========== 主图绘制 ==========
        ct.plot(kind='bar', stacked=True, color=colors_list, ax=ax_main, width=0.7, edgecolor='white', legend=False)

        total_samples = len(data_df)

        def get_text_color(bg_color):
            if isinstance(bg_color, str):
                bg_color = mcolors.to_rgb(bg_color)
            luminance = 0.2126 * bg_color[0] + 0.7152 * bg_color[1] + 0.0722 * bg_color[2]
            return 'white' if luminance < 0.5 else 'black'

        for idx, c in enumerate(ax_main.containers):
            for rect in c:
                height = rect.get_height()
                if height > 0:
                    pct = (height / total_samples) * 100
                    if pct >= 1.0:
                        text_color = get_text_color(colors_list[idx])
                        ax_main.text(rect.get_x() + rect.get_width() / 2,
                                     rect.get_y() + height / 2,
                                     f'{pct:.1f}%',
                                     ha='center', va='center',
                                     color=text_color, fontsize=13)

        ax_main.set_xlabel("CACL Range", fontsize=20)
        ax_main.set_ylabel(f"{level_name.capitalize()} Count", fontsize=20)
        ax_main.tick_params(axis='both', labelsize=18, rotation=0)
        ax_main.grid(axis='y', linestyle='--', alpha=0.3)

        # ========== 放大图绘制 ==========
        if '0.4-1.0' in ct.index:
            ct_zoom = ct.loc[['0.4-1.0']]
            ct_zoom.plot(kind='bar', stacked=True, color=colors_list, ax=ax_zoom, width=0.6, edgecolor='white',
                         legend=False)

            for idx, c in enumerate(ax_zoom.containers):
                for rect in c:
                    height = rect.get_height()
                    if height > 0:
                        pct = (height / total_samples) * 100
                        text_color = get_text_color(colors_list[idx])
                        ax_zoom.text(rect.get_x() + rect.get_width() / 2,
                                     rect.get_y() + height / 2,
                                     f'{pct:.2f}%',
                                     ha='center', va='center',
                                     color=text_color, fontsize=18)

            ax_zoom.set_xlabel("", fontsize=18)
            ax_zoom.set_ylabel("Count", fontsize=18)
            ax_zoom.tick_params(axis='both', labelsize=18)
            ax_zoom.grid(axis='y', linestyle='--', alpha=0.3)
            for label in ax_zoom.get_xticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment('center')

            for spine in ax_zoom.spines.values():
                spine.set_edgecolor('#d62728')
                spine.set_linewidth(2)

        # 图例
        handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_list, BIN_LABELS)]
        ax_main.legend(handles=handles,
                       loc='upper right',
                       bbox_to_anchor=BAR_LEGEND_POS,
                       ncol=1,
                       frameon=False,
                       fontsize=18,
                       title=f"{stack_name}",
                       title_fontsize=18)

        # 连接线
        main_bar_x = 3
        main_bar_top = ct.loc['0.4-1.0'].sum() if '0.4-1.0' in ct.index else 0

        con1 = ConnectionPatch(
            xyA=(main_bar_x + 0.35, main_bar_top), coordsA=ax_main.transData,
            xyB=(-0.3, 0), coordsB=ax_zoom.transData,
            color='#d62728', linestyle='--', linewidth=1.5, alpha=0.6
        )
        fig.add_artist(con1)

        con2 = ConnectionPatch(
            xyA=(main_bar_x + 0.35, 0), coordsA=ax_main.transData,
            xyB=(-0.3, ax_zoom.get_ylim()[1]), coordsB=ax_zoom.transData,
            color='#d62728', linestyle='--', linewidth=1.5, alpha=0.6
        )
        fig.add_artist(con2)

        fname = f"06_stacked_{stack_name.replace(' ', '_')}_{level_name}"
        plt.tight_layout()

        for fmt in OUTPUT_FORMATS:
            plt.savefig(os.path.join(OUTPUT_DIR, f"{fname}.{fmt}"), bbox_inches='tight', dpi=DPI)
        plt.close()

    # 执行堆积图绘制
    plot_stacked_distribution(df_street, 'street', 'D_norm', 'Charging Demand', STACK_COLORS_TIME)
    plot_stacked_distribution(df_station, 'station', 'D_norm', 'Charging Demand', STACK_COLORS_TIME)

    plot_stacked_distribution(df_street, 'street', 'A_i', 'Service Level', STACK_COLORS_AMENITY)
    plot_stacked_distribution(df_station, 'station', 'A_i', 'Service Level', STACK_COLORS_AMENITY)

    print(f"\nAnalysis Completed. Results in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()