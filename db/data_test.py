## Checks final_tracks.db, durations.db, and mp3_files and returns how many track_ids are in each, as well 
## as how many are unique to each. Used to test data after api pull. ##

import os
import sqlite3

MP3_DIR = "/volumes/data/mp3_files"
DB_PATH = "/volumes/data/final_tracks.db"

def get_mp3_track_ids():
    """Returns a set of track IDs from the MP3 file directory, ignoring macOS metadata files."""
    track_ids = set()
    for filename in os.listdir(MP3_DIR):
        if filename.endswith(".mp3") and not filename.startswith("._"):  # Ignore macOS metadata files
            track_id = os.path.splitext(filename)[0]  # Remove .mp3 extension
            track_ids.add(track_id)
    return track_ids

def get_db_track_ids():
    """Returns a set of track IDs from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT track_id FROM tracks")
    track_ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    return track_ids

def get_durations():
    """Returns a set of durations from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT duration FROM tracks")
    durations = {row[0] for row in cursor.fetchall()}
    conn.close()
    return durations

def under_45(x):
    return x < 45


def compare_track_ids():
    """Compares track IDs in the MP3 folder and database, printing statistics."""
    mp3_ids = get_mp3_track_ids()
    db_ids = get_db_track_ids()
    durations = get_durations()

    durations_under_45 = set(filter(under_45, durations))
    shared_ids = mp3_ids & db_ids 
    unique_mp3_ids = mp3_ids - db_ids 
    unique_db_ids = db_ids - mp3_ids 
    
    print(f"mp3_files: {len(mp3_ids)} tracks total")
    print(f"db: {len(db_ids)} tracks total")
    print(f"{len(unique_db_ids)} unique to final_tracks.db")
    print(f"{len(unique_mp3_ids)} unique to mp3_files")
    print(f"{len(shared_ids)} shared")
    print(f"{len(durations_under_45)} durations under 45 seconds")


    
    

if __name__ == "__main__":
    compare_track_ids()
