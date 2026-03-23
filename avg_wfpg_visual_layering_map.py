import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D
import os
import warnings

# ==========================================================
# 1. 核心控制面板 (Parameters)
# ==========================================================
warnings.filterwarnings('ignore')

CSV_PATH = 'data/Raw_data_all_new_select_clustered_gmm_classification_all.csv'
SHP_PATH = 'data/shanghaijiedao/上海市_with_ID_merged.shp'
OUTPUT_DIR = 'W-FPG_Average_VisualLayering_Maps'
OUTPUT_DATA_FILE = os.path.join(OUTPUT_DIR, 'Fig_Spatial_Average_W-FPG_Data.csv')

# --- 绘图窗口定义 ---
TIME_WINDOWS = {
    'LateNight_00-02': [0, 1],
    'MorningPeak_08-11': [8, 9, 10],
    'AfternoonPeak_14-17': [14, 15, 16],
    'EveningPeak_20-23': [20, 21, 22],
    'Whole_Day': list(range(24))
}

# --- 视觉与边界参数 ---
STYLE_CONFIG = {
    'font.family': 'Arial',
    'city_boundary_color': 'grey',
    'city_boundary_width': 1.5,
    'street_edge_color': '#ffffff',
    'street_edge_width': 0.1,

    # 气泡大小范围
    'pos_bubble_min': 10,
    'pos_bubble_max': 1500,

    'neg_bubble_min': 10,
    'neg_bubble_max': 1500,

    # 颜色配置
    'color_pos': '#b91574',  # Shortage (Red/Purple)
    'color_neg': '#62b626',  # Surplus (Green)
    'bubble_alpha': 0.7,

    # 图例坐标位置
    'legend_pos_pos': (0.8, 1),
    'legend_neg_pos': (0.8, 0.49),

    'figsize': (10, 7),
    'dpi': 300
}


# ==========================================================
# 2. 辅助函数与数据计算
# ==========================================================

def safe_read_shapefile(shp_path):
    try:
        return gpd.read_file(shp_path, engine='pyogrio')
    except:
        return gpd.read_file(shp_path)


def calculate_w_fpg_aggregated(df_slice):
    """
    计算聚合后的 W-FPG。
    此时传入的 df_slice 包含7天内特定时间段的所有数据。
    """
    total_events_period = len(df_slice)
    if total_events_period == 0: return None

    # 按站点聚合
    stats = df_slice.groupby('station_id').agg({
        'is_fast_charge_event': 'sum',
        'evdata_vehicle_id': 'count',
        'fast_pile_count': 'first',  # 假设桩数7天不变
        'slow_pile_count': 'first',
        'vehicleposition_longitude': 'mean',
        'vehicleposition_latitude': 'mean'
    }).rename(columns={'is_fast_charge_event': 'delta_s', 'evdata_vehicle_id': 'Delta_s'})

    stats['Sigma_s'] = stats['fast_pile_count'] + stats['slow_pile_count']
    stats = stats[stats['Sigma_s'] > 0]

    # W-FPG 公式
    # 注意：这里的 W_FPG 代表该时间段在7天内的总体平均错配强度
    stats['W_FPG'] = ((stats['delta_s'] / stats['Delta_s']) - (stats['fast_pile_count'] / stats['Sigma_s'])) * (
                stats['Delta_s'] / total_events_period)
    stats['abs_w_fpg'] = stats['W_FPG'].abs()
    return stats


# ==========================================================
# 3. 核心主程序
# ==========================================================

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 3.1 数据加载与预处理
    print("Step 1: Loading Data and Preprocessing...")
    df = pd.read_csv(CSV_PATH)
    df['charge_start_time'] = pd.to_datetime(df['charge_start_time'])
    df['hour'] = df['charge_start_time'].dt.hour

    # 3.2 聚合计算与全局最值扫描
    print("Step 2: Aggregating Data (7-Day Average) & Global Scanning...")

    results_buffer = {}  # 存储计算结果
    max_pos_list = []
    max_neg_list = []

    # 收集导出数据列表
    export_data_list = []

    for win_name, hours in TIME_WINDOWS.items():
        print(f"  Processing window: {win_name}...")
        # 筛选所有日期中，处于该小时段的数据
        df_slice = df[df['hour'].isin(hours)]

        # 计算该时间段的平均指标
        stats = calculate_w_fpg_aggregated(df_slice)

        if stats is not None:
            results_buffer[win_name] = stats

            # 记录用于导出的数据
            stats_export = stats.copy()
            stats_export['Time_Window'] = win_name
            export_data_list.append(stats_export)

            # 记录极值用于后续绘图统一比例尺
            pos_vals = stats[stats['W_FPG'] > 0]['W_FPG']
            neg_vals = stats[stats['W_FPG'] < 0]['W_FPG'].abs()

            if not pos_vals.empty: max_pos_list.append(pos_vals.max())
            if not neg_vals.empty: max_neg_list.append(neg_vals.max())

    # 计算全局最大值 (用于跨图比较)
    GLOBAL_MAX_POS = max(max_pos_list) if max_pos_list else 1.0
    GLOBAL_MAX_NEG = max(max_neg_list) if max_neg_list else 1.0
    print(f"  Global Max Positive W-FPG: {GLOBAL_MAX_POS:.4f}")
    print(f"  Global Max Negative W-FPG: {GLOBAL_MAX_NEG:.4f}")

    # 3.3 导出绘图源数据
    print(f"Step 3: Exporting Source Data to {OUTPUT_DATA_FILE}...")
    if export_data_list:
        df_final_export = pd.concat(export_data_list)
        # 整理列顺序
        cols = ['Time_Window', 'vehicleposition_longitude', 'vehicleposition_latitude',
                'W_FPG', 'abs_w_fpg', 'delta_s', 'Delta_s', 'Sigma_s']
        # 仅保留存在的列
        cols = [c for c in cols if c in df_final_export.columns]
        df_final_export[cols].to_csv(OUTPUT_DATA_FILE, index=True, encoding='utf-8-sig')
    else:
        print("Warning: No data to export.")

    # 3.4 绘图
    gdf_streets = safe_read_shapefile(SHP_PATH)
    gdf_city = gdf_streets.dissolve()
    plt.rcParams['font.family'] = STYLE_CONFIG['font.family']

    print(f"Step 4: Plotting 5 Aggregate Maps...")

    for win_name, stats in results_buffer.items():
        stats = stats.copy()

        # --- 视觉参数计算 (Visual Size) ---
        def get_visual_params(row):
            if row['W_FPG'] >= 0:
                # 正值逻辑
                v_size = (row['abs_w_fpg'] / GLOBAL_MAX_POS) * \
                         (STYLE_CONFIG['pos_bubble_max'] - STYLE_CONFIG['pos_bubble_min']) + \
                         STYLE_CONFIG['pos_bubble_min']
                v_color = STYLE_CONFIG['color_pos']
            else:
                # 负值逻辑
                v_size = (row['abs_w_fpg'] / GLOBAL_MAX_NEG) * \
                         (STYLE_CONFIG['neg_bubble_max'] - STYLE_CONFIG['neg_bubble_min']) + \
                         STYLE_CONFIG['neg_bubble_min']
                v_color = STYLE_CONFIG['color_neg']
            return pd.Series([v_size, v_color], index=['visual_size', 'visual_color'])

        stats[['visual_size', 'visual_color']] = stats.apply(get_visual_params, axis=1)

        # 排序：小球在下，大球在上，避免遮挡
        stats_sorted = stats.sort_values('visual_size', ascending=True)

        # 创建画布
        fig, ax = plt.subplots(figsize=STYLE_CONFIG['figsize'], dpi=STYLE_CONFIG['dpi'])

        # 绘制底图
        gdf_streets.plot(ax=ax, color='#fdfdfd', edgecolor=STYLE_CONFIG['street_edge_color'],
                         linewidth=STYLE_CONFIG['street_edge_width'], zorder=1)
        gdf_city.boundary.plot(ax=ax, color=STYLE_CONFIG['city_boundary_color'],
                               linewidth=STYLE_CONFIG['city_boundary_width'], zorder=2)

        # 绘制气泡
        ax.scatter(stats_sorted['vehicleposition_longitude'],
                   stats_sorted['vehicleposition_latitude'],
                   s=stats_sorted['visual_size'],
                   c=stats_sorted['visual_color'],
                   edgecolors=stats_sorted['visual_color'],
                   alpha=STYLE_CONFIG['bubble_alpha'],
                   zorder=3)

        # 坐标轴设置
        ax.set_xlabel('Longitude (°E)', fontsize=22)
        ax.set_ylabel('Latitude (°N)', fontsize=22)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)

        # --- 图例构建 (固定标尺) ---
        ref_ratios = [0.1, 0.5, 1.0]

        # 图例 1: W-FPG > 0 (红色/紫色)
        pos_refs = [GLOBAL_MAX_POS * r for r in ref_ratios]
        pos_sizes = [(v / GLOBAL_MAX_POS) * (STYLE_CONFIG['pos_bubble_max'] - STYLE_CONFIG['pos_bubble_min']) +
                     STYLE_CONFIG['pos_bubble_min'] for v in pos_refs]
        pos_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLE_CONFIG['color_pos'], alpha=0.5,
                              markersize=np.sqrt(sz), label=f'{v:.3f}') for v, sz in zip(pos_refs, pos_sizes)]
        leg_pos = ax.legend(handles=pos_handles, loc='upper left', bbox_to_anchor=STYLE_CONFIG['legend_pos_pos'],
                            title="W-FPG > 0", title_fontsize=22, fontsize=21, frameon=False, labelspacing=1.2)
        ax.add_artist(leg_pos)

        # 图例 2: W-FPG < 0 (绿色)
        neg_refs = [GLOBAL_MAX_NEG * r for r in ref_ratios]
        neg_sizes = [(v / GLOBAL_MAX_NEG) * (STYLE_CONFIG['neg_bubble_max'] - STYLE_CONFIG['neg_bubble_min']) +
                     STYLE_CONFIG['neg_bubble_min'] for v in neg_refs]
        neg_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLE_CONFIG['color_neg'], alpha=0.5,
                              markersize=np.sqrt(sz), label=f'{v:.3f}') for v, sz in zip(neg_refs, neg_sizes)]
        ax.legend(handles=neg_handles, loc='upper left', bbox_to_anchor=STYLE_CONFIG['legend_neg_pos'],
                  title="W-FPG < 0", title_fontsize=22, fontsize=21, frameon=False, labelspacing=1.2)

        # 保存图片
        save_name_base = f"Average_WFPG_{win_name}"
        plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name_base}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{save_name_base}.pdf"), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name_base}")

    print(f"\nSuccess! 5 aggregate maps and source data saved to: [{OUTPUT_DIR}]")


if __name__ == "__main__":
    main()