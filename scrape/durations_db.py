## Scrapes and creates DB of durations for all songs already stored ## 
## Used to find macrodata about durations for the sake of input data constructions ## 

import sqlite3
import requests
import api_certs
import time
import os

# Configuration
API_URL = "https://api.jamendo.com/v3.0/tracks/"
API_KEY = api_certs.client_id
MP3_DIR = "/volumes/data/mp3_files"
DURATION_DB_PATH = "/volumes/data/durations.db"

# Create duration database
def create_duration_db():
    conn = sqlite3.connect(DURATION_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS durations (
            track_id TEXT PRIMARY KEY,
            duration INTEGER
        )
    """)
    conn.commit()
    conn.close()

# Get track IDs from mp3_files directory
def get_track_ids():
    track_ids = []
    for filename in os.listdir(MP3_DIR):
        if filename.endswith(".mp3") and not filename.startswith("._"):
            track_id = os.path.splitext(filename)[0]
            track_ids.append(track_id)
    return track_ids

# Check if track ID already exists in durations.db
def duration_exists(track_id):
    conn = sqlite3.connect(DURATION_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM durations WHERE track_id = ?", (track_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

# Fetch duration from Jamendo API
def fetch_duration(track_id):
    params = {
        "client_id": API_KEY,
        "format": "json",
        "include": "musicinfo",
        "id": track_id,
        "limit": 1
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0].get("duration", None)
        print(f"No duration found for {track_id}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching duration for {track_id}: {e}")
    return None

# Insert duration into durations.db
def insert_duration(track_id, duration):
    conn = sqlite3.connect(DURATION_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO durations (track_id, duration) VALUES (?, ?)", (track_id, duration))
    conn.commit()
    conn.close()

# Process all track IDs
def process_durations():
    track_ids = get_track_ids()
    total_tracks = len(track_ids)
    processed = 0
    
    for track_id in track_ids:
        if duration_exists(track_id):
            print(f"Track {track_id} already in durations.db. Skipping.")
            continue
        
        duration = fetch_duration(track_id)
        if duration is not None:
            insert_duration(track_id, duration)
            processed += 1
            print(f"{processed}/{total_tracks} tracks processed.")
        
        time.sleep(1)  # Prevent API rate limits

if __name__ == "__main__":
    create_duration_db()
    process_durations()
