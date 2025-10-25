import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans

sns.set(style='whitegrid')

DATA_PATH = 'data/sleep_data.csv'
OUT_DIR = 'outputs'

os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=['sleep_start','sleep_end'])
    # ensure duration
    if 'duration_min' not in df.columns:
        df['duration_min'] = (df['sleep_end'] - df['sleep_start']).dt.total_seconds() / 60
    return df

def preprocess(df):
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['bedtime_hour'] = df['sleep_start'].dt.hour + df['sleep_start'].dt.minute/60
    df['wake_hour'] = df['sleep_end'].dt.hour + df['sleep_end'].dt.minute/60
    return df

def summary_stats(df):
    summary = df.groupby('user_id').agg(
        nights=('date','nunique'),
        avg_duration=('duration_min','mean'),
        std_duration=('duration_min','std'),
        avg_bedtime=('bedtime_hour','mean'),
        avg_quality=('sleep_quality','mean')
    ).reset_index()
    summary.to_csv(os.path.join(OUT_DIR,'user_summary.csv'), index=False)
    return summary

def plot_distributions(df):
    plt.figure(figsize=(8,4))
    sns.histplot(df['duration_min'], bins=30)
    plt.xlabel('Duration (min)')
    plt.title('Distribution of Sleep Duration')
    plt.savefig(os.path.join(OUT_DIR,'duration_hist.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    sns.boxplot(x='user_id', y='duration_min', data=df)
    plt.xticks(rotation=90)
    plt.title('Duration by User')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'duration_by_user.png'))
    plt.close()

def plot_bedtime_heatmap(df):
    df['weekday'] = pd.to_datetime(df['date']).dt.day_name()
    pivot = df.pivot_table(index='weekday', columns=pd.cut(df['bedtime_hour'], bins=24, labels=np.arange(24)), values='duration_min', aggfunc='mean')
    # order weekdays
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = pivot.reindex(order)
    plt.figure(figsize=(12,5))
    sns.heatmap(pivot, cmap='viridis', cbar_kws={'label':'avg duration (min)'})
    plt.title('Avg sleep duration by bedtime hour and weekday')
    plt.xlabel('Bedtime hour (binned)')
    plt.savefig(os.path.join(OUT_DIR,'bedtime_heatmap.png'))
    plt.close()

def sleep_stage_proportions(df):
    # if rem/light/deep exist
    if all(c in df.columns for c in ['rem_min','light_min','deep_min']):
        stages = df[['rem_min','light_min','deep_min']].sum()
        stages = stages / stages.sum()
        stages.plot(kind='pie', autopct='%1.1f%%', ylabel='')
        plt.title('Sleep stage proportion (total)')
        plt.savefig(os.path.join(OUT_DIR,'stage_proportions.png'))
        plt.close()

def clustering(summary, n_clusters=3):
    features = summary[['avg_duration','std_duration','avg_bedtime','avg_quality']].fillna(0)
    km = KMeans(n_clusters=n_clusters, random_state=0)
    labels = km.fit_predict(features)
    summary['cluster'] = labels
    summary.to_csv(os.path.join(OUT_DIR,'user_summary_with_clusters.csv'), index=False)
    # plot clusters
    plt.figure(figsize=(7,5))
    sns.scatterplot(x='avg_bedtime', y='avg_duration', hue='cluster', data=summary, palette='tab10')
    plt.xlabel('Avg bedtime hour')
    plt.ylabel('Avg duration (min)')
    plt.title('User clusters by bedtime and duration')
    plt.savefig(os.path.join(OUT_DIR,'clusters.png'))
    plt.close()

def run_all(path=DATA_PATH):
    df = load_data(path)
    df = preprocess(df)
    summary = summary_stats(df)
    plot_distributions(df)
    plot_bedtime_heatmap(df)
    sleep_stage_proportions(df)
    clustering(summary)
    print('Analysis complete. Outputs saved in', OUT_DIR)

if __name__ == '__main__':
    run_all()