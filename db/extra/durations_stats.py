## Get various metrics from durations.db ## 

import sqlite3
import statistics as stat

DUR_PATH = "/volumes/data/durations.db"

def get_duration_track_ids():
    """Returns a set of track IDs from the durations database"""
    conn = sqlite3.connect(DUR_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT duration FROM durations")
    durations = [row[0] for row in cursor.fetchall()]
    conn.close()
    return durations

durations = get_duration_track_ids()
def lengths_under_46():
    counter = 0
    for duration in durations:
        if duration < 46:
            counter += 1
    return counter

max_duration, min_duration = max(durations), min(durations)
average_duration, median_duration = stat.mean(durations), stat.median(durations)
stdev = stat.stdev(durations)
under46 = lengths_under_46()

print(f"Max: {max_duration}")
print(f"Min: {min_duration}")
print(f"Average: {average_duration}")
print(f"Median: {median_duration}")
print(f"StDev: {stdev}")
print(f"{under46} tracks under 46 seconds.")