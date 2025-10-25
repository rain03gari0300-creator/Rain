import csv
import random
from datetime import datetime, timedelta
import os

os.makedirs('data', exist_ok=True)

def random_time(date, hour_mean, hour_sd):
    base = datetime.combine(date, datetime.min.time())
    minutes = int(random.gauss(hour_mean*60, hour_sd*60))
    return base + timedelta(minutes=minutes)

def generate(n_users=10, days=30, out_file='data/sleep_data.csv'):
    header = ['user_id','date','sleep_start','sleep_end','duration_min','sleep_quality','rem_min','light_min','deep_min','awakenings','avg_heart_rate']
    rows = []
    start_date = datetime.today().date() - timedelta(days=days)
    for u in range(1, n_users+1):
        for d in range(days):
            date = start_date + timedelta(days=d)
            # simulate bedtime around 23:00-01:00
            bedtime = random_time(date, hour_mean=23.5 + random.choice([-1,0,1])*0.3, hour_sd=0.75)
            # duration between 300 and 540 minutes
            duration = max(240, int(random.gauss(420, 60)))
            wake = bedtime + timedelta(minutes=duration)
            rem = max(10, int(duration * random.uniform(0.15, 0.22)))
            deep = max(20, int(duration * random.uniform(0.10, 0.20)))
            light = duration - rem - deep
            awakenings = max(0, int(random.expovariate(1/0.5)))
            hr = int(random.gauss(60 - u%5, 4))
            rows.append([f'user_{u}', date.isoformat(), bedtime.isoformat(), wake.isoformat(), duration, round(random.uniform(0.5,1.0),2), rem, light, deep, awakenings, hr])

    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

if __name__ == '__main__':
    generate(n_users=20, days=60)