import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, pi
import warnings
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

INPUT_FILE = 'data/Raw_data_all_new_select.csv'
OUTPUT_DIR = 'output/'
OUTPUT_FEATURES_FILE = 'V2-Simple_01_driver_features_statistics.csv'
OUTPUT_CLASSIFICATION_FILE = 'V2-Simple_02_regularity_classification.csv'

MIN_DAYS_WITH_CHARGE = 4
MIN_CHARGE_EVENTS = 5

ABSOLUTE_THRESHOLDS = {
    'High': 0.75,
    'Medium': 0.5
}


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def calculate_time_regularity(series_hour):
    if series_hour.empty or series_hour.nunique() == 1:
        return 1.0
    theta = (series_hour / 24) * 2 * pi
    C = np.cos(theta).mean()
    S = np.sin(theta).mean()
    R_bar = np.sqrt(C ** 2 + S ** 2)
    return R_bar


def calculate_space_regularity(group):
    if group.empty or len(group) == 1:
        return 0.0
    centroid_lon = group['vehicleposition_longitude'].mean()
    centroid_lat = group['vehicleposition_latitude'].mean()
    distances = [
        haversine(centroid_lon, centroid_lat, row['vehicleposition_longitude'], row['vehicleposition_latitude'])
        for _, row in group.iterrows()
    ]
    return np.mean(distances)


def get_entropy_weights(R_matrix):
    P = R_matrix / R_matrix.sum(axis=0)
    P = P + 1e-10
    k = 1 / np.log(len(P))
    E = -k * (P * np.log(P)).sum(axis=0)
    D = 1 - E
    W = D / D.sum()
    return W


def classify_absolute(r_score, thresholds=ABSOLUTE_THRESHOLDS):
    if r_score >= thresholds['High']:
        return 'High'
    elif r_score >= thresholds['Medium']:
        return 'Medium'
    else:
        return 'Low'


def load_and_clean_data(filepath):
    print("-" * 60)
    print("Step 1: Data Loading and Cleaning")
    print("-" * 60)

    df = pd.read_csv(filepath)
    print(f"Raw data size: {len(df)}")

    df['charge_start_time'] = pd.to_datetime(df['charge_start_time'])
    df['charge_end_time'] = pd.to_datetime(df['charge_end_time'])

    key_cols = ['evdata_vehicle_id', 'charge_start_time', 'start_charge_hour',
                'vehicleposition_longitude', 'vehicleposition_latitude',
                'soc_difference', 'initial_soc']
    df = df.dropna(subset=key_cols)

    df = df[df['soc_difference'] > 0]
    df = df[df['vehicleposition_longitude'] != 0]

    print(f"Cleaned data size: {len(df)}")
    df['charge_date'] = df['charge_start_time'].dt.date
    return df


def filter_valid_drivers(df, min_days=MIN_DAYS_WITH_CHARGE, min_events=MIN_CHARGE_EVENTS):
    print("\n" + "-" * 60)
    print("Step 2: Driver Filtering")
    print("-" * 60)

    driver_stats = df.groupby('evdata_vehicle_id').agg({
        'charge_date': 'nunique',
        'charge_start_time': 'count'
    }).rename(columns={'charge_date': 'charge_days', 'charge_start_time': 'charge_events'})

    valid_drivers = driver_stats[
        (driver_stats['charge_days'] >= min_days) &
        (driver_stats['charge_events'] >= min_events)
        ].index

    df_filtered = df[df['evdata_vehicle_id'].isin(valid_drivers)].copy()

    print(f"Original unique drivers: {df['evdata_vehicle_id'].nunique()}")
    print(f"Valid drivers after filtering: {len(valid_drivers)}")
    print(f"Retained data size: {len(df_filtered)}")

    return df_filtered


def extract_features_v2_simple(df):
    print("\n" + "-" * 60)
    print("Step 3: Feature Extraction")
    print("-" * 60)

    df_grouped = df.groupby('evdata_vehicle_id')

    print("Calculating aggregate features...")
    features_agg = df_grouped.agg(
        total_charge_events=pd.NamedAgg(column='charge_start_time', aggfunc='count'),
        charge_days=pd.NamedAgg(column='charge_date', aggfunc='nunique'),
        R_strategy_soc_diff_std=pd.NamedAgg(column='soc_difference', aggfunc='std'),
    )
    features_agg['R_strategy_soc_diff_std'] = features_agg['R_strategy_soc_diff_std'].fillna(0)

    print("Calculating time regularity...")
    R_time = df_grouped['start_charge_hour'].apply(calculate_time_regularity)
    R_time.name = 'R_time'

    print("Calculating space regularity...")
    R_space_dist = df_grouped.apply(calculate_space_regularity)
    R_space_dist.name = 'R_space_centroid_dist_km'

    print("Merging features...")
    features_df = pd.concat([features_agg, R_time, R_space_dist], axis=1)

    print(f"Feature extraction completed for {len(features_df)} drivers")
    return features_df


def calculate_regularity_and_classify_v2_simple(features_df):
    print("\n" + "-" * 60)
    print("Step 4: Regularity Calculation and Classification")
    print("-" * 60)

    result_df = features_df.copy()

    print("Normalizing metrics (Min-Max)...")

    def normalize(col):
        if (col.max() - col.min()) == 0:
            return np.ones_like(col)
        return (col - col.min()) / (col.max() - col.min())

    result_df['R_time_norm'] = normalize(result_df['R_time'])
    result_df['R_space_norm'] = 1 - normalize(result_df['R_space_centroid_dist_km'])
    result_df['R_strategy_norm'] = 1 - normalize(result_df['R_strategy_soc_diff_std'])

    print("Calculating entropy weights...")

    metrics_matrix = result_df[['R_time_norm', 'R_space_norm', 'R_strategy_norm']].values

    weights = get_entropy_weights(metrics_matrix)

    W_time_final = weights[0]
    W_space_final = weights[1]
    W_strategy_final = weights[2]

    print(f"Weights calculated:")
    print(f"R_time weight: {W_time_final:.3f}")
    print(f"R_space weight: {W_space_final:.3f}")
    print(f"R_strategy weight: {W_strategy_final:.3f}")

    print("Calculating final regularity score...")

    result_df['R_strategy'] = result_df['R_strategy_norm']

    result_df['R_total'] = (
            W_time_final * result_df['R_time_norm'] +
            W_space_final * result_df['R_space_norm'] +
            W_strategy_final * result_df['R_strategy']
    )

    print(f"Applying classification thresholds ({ABSOLUTE_THRESHOLDS})...")

    result_df['Overall_Class'] = result_df['R_total'].apply(classify_absolute)
    result_df['Time_Class'] = result_df['R_time_norm'].apply(classify_absolute)
    result_df['Space_Class'] = result_df['R_space_norm'].apply(classify_absolute)
    result_df['Strategy_Class'] = result_df['R_strategy'].apply(classify_absolute)

    print("\n" + "-" * 30)
    print("Classification Statistics:")
    print(result_df['Overall_Class'].value_counts(normalize=True).sort_index())
    print("-" * 30)

    return result_df


def save_results_v2(features_df, classification_df, output_dir):
    print("\n" + "-" * 60)
    print("Step 5: Saving Results")
    print("-" * 60)

    os.makedirs(output_dir, exist_ok=True)

    output_file1 = os.path.join(output_dir, OUTPUT_FEATURES_FILE)
    features_df.to_csv(output_file1, index=True, encoding='utf-8-sig')
    print(f"Saved: {output_file1}")

    cols_to_save = ['Overall_Class', 'Time_Class', 'Space_Class', 'Strategy_Class']
    output_file2 = os.path.join(output_dir, OUTPUT_CLASSIFICATION_FILE)
    classification_df[cols_to_save].to_csv(output_file2, index=True, encoding='utf-8-sig')
    print(f"Saved: {output_file2}")


def main():
    print("\n" + "-" * 60)
    print("Driver Charging Regularity Analysis")
    print("-" * 60)

    df = load_and_clean_data(INPUT_FILE)

    df_filtered = filter_valid_drivers(df)

    if len(df_filtered) == 0:
        print("\n" + "!" * 60)
        print("Error: No drivers matched the filtering criteria.")
        print(f"Check parameters: MIN_DAYS={MIN_DAYS_WITH_CHARGE}, MIN_EVENTS={MIN_CHARGE_EVENTS}")
        print("!" * 60)
        return

    features_df = extract_features_v2_simple(df_filtered)

    classification_df = calculate_regularity_and_classify_v2_simple(features_df)

    save_results_v2(features_df=classification_df,
                    classification_df=classification_df,
                    output_dir=OUTPUT_DIR)

    print("\n" + "-" * 60)
    print("Analysis Completed Successfully")
    print("-" * 60)
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print("Output Files:")
    print(f"1. {OUTPUT_FEATURES_FILE}")
    print(f"2. {OUTPUT_CLASSIFICATION_FILE}")
    print("\n" + "-" * 60)


if __name__ == '__main__':
    main()