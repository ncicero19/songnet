## Final API Procedure. Processes data completely and loads into final_tracks.db and mp3_files ##
## Stores only track_id and relevant tags, mapped to a common theme ##
## Resistant to duplicates or network connectivity issues ## 
## The bug had been creating a different table called mood_tracks with track_id and mood. Still need to remove ##

import os
import requests
import sqlite3
import api_certs
import alltags
import tag_map
import time

# Configuration (more parameters in fetch_tracks())
API_URL = "https://api.jamendo.com/v3.0/tracks/"
API_KEY = api_certs.client_id
SAVE_DIR = "/volumes/data/mp3_files"
DB_PATH = "/volumes/data/final_tracks.db"
LIMIT = 200  # Max items per API request
MAX_TRACKS = 5004  # Target number of tracks in final_tracks.db
TAG_LIST = set(alltags.alltags)  # Convert to set for faster lookup


# Ensure directories exist
os.makedirs(SAVE_DIR, exist_ok=True)

def create_db():
    """Creates an SQLite database to store track metadata."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY,
            track_id TEXT UNIQUE,
            mood_tags TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_existing_track_count():
    """Returns the current number of tracks in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tracks")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def track_exists(track_id):
    """Checks if a track ID already exists in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM tracks WHERE track_id = ?", (track_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def download_mp3(track_id, url):
    """Downloads an MP3 file and saves it locally."""
    file_path = os.path.join(SAVE_DIR, f"{track_id}.mp3")
    
    # Check for duplicates
    if os.path.exists(file_path):
        print(f"{track_id} already exists. Skip.")
        return None  # Return None to prevent decreasing remaining_tracks
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {track_id}")
            return file_path
        else:
            print(f"Failed to download {track_id}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {track_id}: {e}")
        return None

def insert_track(track_id, mood_tags):
    """Inserts track metadata into the database."""
    if not mood_tags:
        print(f"{track_id} skipped -- bad data")
        return False  # Skip tracks with no matching tags
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO tracks (track_id, mood_tags)
        VALUES (?, ?)
    """, (track_id, ",".join(mood_tags)))
    conn.commit()
    inserted = cursor.rowcount > 0
    conn.close()
    return inserted

def fetch_tracks(offset=0):
    """Fetches track data from the Jamendo API in batches."""
    ## Consult with the Jamendo API webpage for additional params
    params = {
        "client_id": API_KEY,
        "format": "json",
        "limit": min(LIMIT, MAX_TRACKS - get_existing_track_count()),
        "offset": offset, ## How many tracks in to start from 
        "include": "musicinfo", ## What info is included 
        "audioformat": "mp32", ## Audio quality 
        "order": "downloads_total" ## How the results are sorted
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        if response.status_code != 200:
            print(f"Error fetching data: {response.text}")
            return []
        return response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

def process_tracks():
    """Fetches and processes tracks in batches until MAX_TRACKS is reached."""
    existing_tracks = get_existing_track_count()
    remaining_tracks = MAX_TRACKS - existing_tracks
    offset = 0
    
    while remaining_tracks > 0:
        tracks = fetch_tracks(offset)
        if not tracks:
            break  # Stop when no more tracks are returned
        
        for track in tracks:
            if remaining_tracks <= 0:
                break
            
            track_id = track.get("id")
            if track_exists(track_id):
                print(f"{track_id} already exists. Skip.")
                continue
            
            mood_tags = track.get("musicinfo", {}).get("tags", {}).get("vartags", []) or []
            audio_url = track.get("audiodownload")
            
            if not audio_url:
                print(f"Skip {track_id} -- no download link")
                continue
            
            mapped_moods = set()
            for tag in mood_tags:
                for mood, keywords in tag_map.tag_map.items():
                    if tag in keywords:
                        mapped_moods.add(mood)
                        
            
            if not mapped_moods:
                print(f"Skipping {track_id} -- bad tags")
                continue
            
            file_path = download_mp3(track_id, audio_url)
            if file_path:
                if insert_track(track_id, list(mapped_moods)):
                    remaining_tracks -= 1
                    print(f"{remaining_tracks} remaining.")
        
        offset += LIMIT  # Move to the next batch
        time.sleep(5)  # Prevent API rate limiting

if __name__ == "__main__":
    create_db()
    process_tracks()
