## Cleans tracks.db keeping only tracks with tags in tag_map.py and only track ID and tags ##

import sqlite3
import tag_map  # Import the tag mapping dictionary

# Paths
DB_PATH = "/volumes/data/tracks.db"
FILTERED_DB_PATH = "/volumes/data/goodtag_tracks.db"

# Flatten the tag_map to get all valid tags
valid_tags = set(tag for tags in tag_map.tag_map.values() for tag in tags)

def create_filtered_db():
    """Creates a new SQLite database to store filtered tracks."""
    conn = sqlite3.connect(FILTERED_DB_PATH)
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

def filter_tracks():
    """Filters tracks based on approved tags and saves them to a new database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT track_id, mood_tags FROM tracks")
    tracks = cursor.fetchall()
    conn.close()
    
    filtered_tracks = []
    for track in tracks:
        track_id, mood_tags = track
        
        # Convert stored tags into a list
        track_moods = set(mood_tags.split(",")) if mood_tags else set()
        
        # Keep track if at least one tag is valid
        valid_moods = list(track_moods & valid_tags)
        if valid_moods:
            filtered_tracks.append((track_id, ",".join(valid_moods)))
    
    # Insert filtered tracks into new DB
    conn = sqlite3.connect(FILTERED_DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO tracks (track_id, mood_tags)
        VALUES (?, ?)
    """, filtered_tracks)
    conn.commit()
    conn.close()
    
    print(f"Filtered database created with {len(filtered_tracks)} tracks.")

if __name__ == "__main__":
    create_filtered_db()
    filter_tracks()
