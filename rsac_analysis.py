import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
import warnings
import os
from itertools import combinations

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.grid'] = False
plt.rcParams['lines.linewidth'] = 2.5

warnings.filterwarnings('ignore')


def calculate_rsac_core(df, alpha=2.5, beta=0.5, epsilon=1.0):
    print("=" * 80)
    print("Step 1: Calculating RSAC Index (Regularity-Service Accessibility Coupling Index)")
    print("=" * 80)

    for col in ['Time_Class', 'Space_Class', 'Strategy_Class']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['r_tau'] = df['Time_Class'] / 2.0
    df['r_sigma'] = df['Space_Class'] / 2.0
    df['r_pi'] = df['Strategy_Class'] / 2.0

    eta_city = df['wait_time'].mean()
    print(f"\n   City average wait time (eta_city): {eta_city:.2f} min")

    driver_stats = df.groupby('evdata_vehicle_id').agg({
        'r_tau': 'mean',
        'r_sigma': 'mean',
        'r_pi': 'mean',
        'wait_time': 'mean',
        'station_id': 'count',
        'Time_Class': 'first',
        'Space_Class': 'first',
        'Strategy_Class': 'first'
    }).rename(columns={'wait_time': 'eta_j', 'station_id': 'total_charges'})

    driver_stats['r_bar'] = (driver_stats['r_tau'] + driver_stats['r_sigma'] + driver_stats['r_pi']) / 3.0

    driver_stats['r_norm'] = np.sqrt(
        driver_stats['r_tau'] ** 2 +
        driver_stats['r_sigma'] ** 2 +
        driver_stats['r_pi'] ** 2
    )

    driver_stats['cosine_sim'] = (np.sqrt(3) * driver_stats['r_bar']) / (driver_stats['r_norm'] + 1e-9)
    driver_stats['cosine_sim'] = driver_stats['cosine_sim'].clip(upper=1.0)

    term1 = np.power(driver_stats['r_bar'], alpha)
    term2 = np.power(driver_stats['cosine_sim'], beta)
    term3 = (eta_city + epsilon) / (driver_stats['eta_j'] + epsilon)

    driver_stats['Theta_j'] = term1 * term2 * term3

    driver_stats['Status'] = pd.cut(driver_stats['Theta_j'],
                                    bins=[-np.inf, 0.8, 1.0, np.inf],
                                    labels=['Discriminated', 'Fair', 'Prioritized'])

    print(f"   Calculation complete, total drivers: {len(driver_stats)}")
    print("=" * 80)

    return driver_stats


class RSAC_Visualizer:
    def __init__(self, driver_df, raw_df, output_dir='output_plots'):
        self.driver_df = driver_df
        self.raw_df = raw_df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_fig(self, fig, name, engine='matplotlib', save_formats=['png', 'pdf']):
        for fmt in save_formats:
            path = os.path.join(self.output_dir, f"{name}.{fmt}")
            try:
                if engine == 'plotly':
                    if fmt.lower() == 'html':
                        fig.write_html(path)
                    elif fmt.lower() == 'png':
                        fig.write_image(path, width=1200, height=900)
                    elif fmt.lower() == 'pdf':
                        fig.write_image(path, width=1200, height=900)
                    else:
                        continue
                else:  # matplotlib
                    plt.savefig(path, dpi=300, bbox_inches='tight', format=fmt)
                print(f"   Saved: {name}.{fmt}")
            except Exception as e:
                print(f"   Failed to save {name}.{fmt}: {str(e)}")

        if engine == 'matplotlib':
            plt.close()

    def plot_3d_sphere(self, save_formats=['png', 'pdf']):
        print("\n" + "=" * 80)
        print("Drawing Chart 1: 3D Behavioral Sphere (All Data Points)")
        print("=" * 80)

        plot_df = self.driver_df.copy()

        # Output Source Data
        csv_path = os.path.join(self.output_dir, "Fig_RSAC_3D_Sphere_Data.csv")
        plot_df.to_csv(csv_path, index=True, encoding='utf-8-sig')
        print(f"   Source data saved: {csv_path}")

        print(f"   Total data points to plot: {len(plot_df)}")

        fig = px.scatter_3d(
            plot_df,
            x='r_tau',
            y='r_sigma',
            z='r_pi',
            color='Theta_j',
            size='total_charges',
            color_continuous_scale='Turbo',
            opacity=0.7,
            title="3D Behavioral Regularity Sphere (Complete Dataset)",
            labels={
                'r_tau': 'Time Regularity',
                'r_sigma': 'Space Regularity',
                'r_pi': 'Strategy Regularity',
                'Theta_j': 'RSAC Index'
            }
        )

        fig.add_trace(go.Scatter3d(
            x=[0, 1],
            y=[0, 1],
            z=[0, 1],
            mode='lines+text',
            text=['Origin', 'Ideal(1,1,1)'],
            line=dict(color='red', width=5),
            name='Ideal Vector',
            textposition='top center',
            textfont=dict(size=14, color='red')
        ))

        fig.update_layout(
            font=dict(family="Arial", size=14),
            scene=dict(
                xaxis_title='Time Regularity (r_tau)',
                yaxis_title='Space Regularity (r_sigma)',
                zaxis_title='Strategy Regularity (r_pi)',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                zaxis=dict(range=[0, 1])
            ),
            width=1200,
            height=900
        )

        self.save_fig(fig, "01_3D_Sphere", engine='plotly', save_formats=save_formats)
        print("=" * 80)

    def plot_cdf_enhanced(self, save_formats=['png', 'pdf']):
        print("\n" + "=" * 80)
        print("Drawing Chart 2: Enhanced CDF Curve with Comprehensive Statistics")
        print("=" * 80)

        labels = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        self.driver_df['freq_bin'] = pd.qcut(
            self.driver_df['total_charges'],
            q=5,
            labels=labels,
            duplicates='drop'
        )

        # Output Source Data
        csv_path = os.path.join(self.output_dir, "Fig_RSAC_CDF_Data.csv")
        self.driver_df.to_csv(csv_path, index=True, encoding='utf-8-sig')
        print(f"   Source data saved: {csv_path}")

        self._print_detailed_cdf_statistics(labels)

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        for i, label in enumerate(labels):
            subset = self.driver_df[self.driver_df['freq_bin'] == label]['Theta_j'].dropna().sort_values()
            n = len(subset)

            if n == 0:
                continue

            y = np.arange(1, n + 1) / n
            x = subset.values

            ax.plot(x, y, label=label, color=colors[i], linewidth=2.5)

            if i in [0, 4]:
                alpha_conf = 0.05
                epsilon_dkw = np.sqrt(np.log(2 / alpha_conf) / (2 * n))
                y_lower = np.maximum(y - epsilon_dkw, 0)
                y_upper = np.minimum(y + epsilon_dkw, 1)
                ax.fill_between(x, y_lower, y_upper, color=colors[i], alpha=0.15, linewidth=0)

        for q in [0.25, 0.50, 0.75]:
            ax.axhline(q, color='gray', linestyle=':', linewidth=1, alpha=0.6)
            ax.text(ax.get_xlim()[1] * 0.02, q + 0.01, f'{q * 100}%', color='gray', fontsize=22)

        ax.axvline(1.0, color='black', linestyle='--', linewidth=2)
        ax.text(1.02, 0.05, 'Fairness Baseline (1.0)', rotation=90, va='bottom', fontsize=22)

        ax.axvline(0.8, color='#d62728', linestyle='--', linewidth=2)
        ax.text(0.82, 0.05, 'Discrimination (0.8)', rotation=90, va='bottom', color='#d62728', fontsize=22)

        ax.set_xlabel("RSAC Index")
        ax.set_ylabel("Cumulative Probability")

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        ax.tick_params(axis='x', which='both', top=False, direction='out', length=6, width=1.5)
        ax.tick_params(axis='y', which='both', right=False, direction='out', length=6, width=1.5)

        ax.legend(title="Charging Frequency", loc='lower right', frameon=False,
                  fontsize=22, title_fontsize=22)

        self.save_fig(plt, "04_CDF_Enhanced", engine='matplotlib', save_formats=save_formats)
        print("=" * 80)

    def _print_detailed_cdf_statistics(self, labels):
        print("\n" + "─" * 80)
        print("DETAILED CDF STATISTICS REPORT".center(80))
        print("─" * 80)

        print("\n[1. Quintile Group Statistics]")
        print("─" * 80)
        print(
            f"{'Group':<20} {'Count':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'Max':>10}")
        print("─" * 80)

        group_stats = {}
        for label in labels:
            subset = self.driver_df[self.driver_df['freq_bin'] == label]['Theta_j'].dropna()
            group_stats[label] = {
                'data': subset,
                'count': len(subset),
                'mean': subset.mean(),
                'std': subset.std(),
                'min': subset.min(),
                'q25': subset.quantile(0.25),
                'q50': subset.quantile(0.50),
                'q75': subset.quantile(0.75),
                'max': subset.max()
            }

            stats = group_stats[label]
            print(f"{label:<20} {stats['count']:>8} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                  f"{stats['min']:>10.4f} {stats['q25']:>10.4f} {stats['q50']:>10.4f} "
                  f"{stats['q75']:>10.4f} {stats['max']:>10.4f}")

        print("\n[2. Key Threshold Analysis]")
        print("─" * 80)
        thresholds = [0.8, 1.0, 1.2]

        for label in labels:
            data = group_stats[label]['data']
            print(f"\n  {label}:")
            for thresh in thresholds:
                below_count = (data < thresh).sum()
                below_pct = below_count / len(data) * 100
                print(f"    - Below {thresh:.1f}: {below_count:>5} drivers ({below_pct:>6.2f}%)")

        print("\n[3. Pairwise Statistical Significance Tests (KS Test)]")
        print("─" * 80)
        print(f"{'Comparison':<30} {'KS Statistic':>15} {'P-value':>15} {'Significance':>15}")
        print("─" * 80)

        for (label1, label2) in combinations(labels, 2):
            data1 = group_stats[label1]['data']
            data2 = group_stats[label2]['data']

            if len(data1) > 0 and len(data2) > 0:
                ks_stat, p_val = scipy_stats.ks_2samp(data1, data2)
                significance = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
                print(f"{label1} vs {label2:<15} {ks_stat:>15.4f} {p_val:>15.4e} {significance:>15}")

        print("\n  Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")

        print("\n[4. One-Way ANOVA Test]")
        print("─" * 80)

        groups_data = [group_stats[label]['data'].values for label in labels if len(group_stats[label]['data']) > 0]
        if len(groups_data) > 1:
            f_stat, p_val = scipy_stats.f_oneway(*groups_data)
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  P-value:     {p_val:.4e}")
            if p_val < 0.001:
                print(f"  Result:      Highly significant difference among groups (p < 0.001) ***")
            elif p_val < 0.05:
                print(f"  Result:      Significant difference among groups (p < 0.05) *")
            else:
                print(f"  Result:      No significant difference among groups (p >= 0.05)")

        print("\n[5. Effect Size Analysis (Q1 vs Q5)]")
        print("─" * 80)

        q1_data = group_stats[labels[0]]['data']
        q5_data = group_stats[labels[-1]]['data']

        if len(q1_data) > 0 and len(q5_data) > 0:
            mean_diff = q5_data.mean() - q1_data.mean()
            pooled_std = np.sqrt((q1_data.std() ** 2 + q5_data.std() ** 2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            print(f"  Mean Difference (Q5 - Q1): {mean_diff:>10.4f}")
            print(f"  Pooled Std Dev:             {pooled_std:>10.4f}")
            print(f"  Cohen's d:                  {cohens_d:>10.4f}")

            if abs(cohens_d) < 0.2:
                effect_interpretation = "Negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "Small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "Medium"
            else:
                effect_interpretation = "Large"
            print(f"  Interpretation:             {effect_interpretation}")

        print("\n[6. Inequality Metrics]")
        print("─" * 80)

        all_theta = self.driver_df['Theta_j'].dropna().sort_values().values
        n = len(all_theta)
        if n > 0:
            gini = (2 * np.sum((np.arange(1, n + 1)) * all_theta)) / (n * np.sum(all_theta)) - (n + 1) / n
            print(f"  Gini Coefficient: {gini:.4f}")
            print(f"    (0 = perfect equality, 1 = perfect inequality)")

            cv = self.driver_df['Theta_j'].std() / self.driver_df['Theta_j'].mean()
            print(f"  Coefficient of Variation: {cv:.4f}")

        print("\n[7. Fairness Assessment]")
        print("─" * 80)

        total_drivers = len(self.driver_df)
        discriminated = (self.driver_df['Theta_j'] < 0.8).sum()
        fair = ((self.driver_df['Theta_j'] >= 0.8) & (self.driver_df['Theta_j'] <= 1.0)).sum()
        prioritized = (self.driver_df['Theta_j'] > 1.0).sum()

        print(f"  Total Drivers:                {total_drivers:>8}")
        print(f"  Discriminated (Theta < 0.8):     {discriminated:>8} ({discriminated / total_drivers * 100:>6.2f}%)")
        print(f"  Fair Coupling (0.8 <= Theta <= 1): {fair:>8} ({fair / total_drivers * 100:>6.2f}%)")
        print(f"  Prioritized (Theta > 1.0):       {prioritized:>8} ({prioritized / total_drivers * 100:>6.2f}%)")

        print("\n" + "─" * 80)
        print("END OF STATISTICS REPORT".center(80))
        print("─" * 80 + "\n")


def main():
    CSV_PATH = 'data/Raw_data_all_new_select_clustered_gmm_classification_all.csv'

    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        print("Trying alternative path...")
        CSV_PATH = 'data/Raw_data_all_new_select_clustered_gmm_classification.csv'

        if not os.path.exists(CSV_PATH):
            print("Generating DUMMY data for testing...")
            N = 2000
            raw_df = pd.DataFrame({
                'evdata_vehicle_id': np.random.randint(0, 200, N),
                'Time_Class': np.random.randint(0, 3, N),
                'Space_Class': np.random.randint(0, 3, N),
                'Strategy_Class': np.random.randint(0, 3, N),
                'wait_time': np.random.exponential(15, N),
                'station_id': np.random.randint(0, 50, N),
            })
        else:
            raw_df = pd.read_csv(CSV_PATH)
    else:
        raw_df = pd.read_csv(CSV_PATH)

    print(f"\nLoaded dataset: {len(raw_df)} charging events")

    driver_df = calculate_rsac_core(raw_df)

    viz = RSAC_Visualizer(driver_df, raw_df)

    save_formats = ['png', 'pdf']

    viz.plot_3d_sphere(save_formats=save_formats)
    viz.plot_cdf_enhanced(save_formats=save_formats)

    print("\n" + "=" * 80)
    print("All visualizations and statistics generated successfully!")
    print(f"Output formats: PNG and PDF")
    print(f"Output directory: {os.path.abspath(viz.output_dir)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()