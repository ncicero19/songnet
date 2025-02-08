import sqlite3
import tag_map  # Import the tag mapping dictionary

# Paths
INPUT_DB_PATH = "/volumes/data/goodtag_tracks.db"
OUTPUT_DB_PATH = "/volumes/data/final_tracks.db"

# Create a reversed mapping from tag to mood
reversed_tag_map = {}
for mood, tags in tag_map.tag_map.items():
    for tag in tags:
        reversed_tag_map[tag] = reversed_tag_map.get(tag, set()) | {mood}

def create_mood_db():
    """Creates a new SQLite database to store mood-based tracks."""
    print("Creating mood database...")
    conn = sqlite3.connect(OUTPUT_DB_PATH)
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
    print(f"Mood database created at {OUTPUT_DB_PATH}")

def convert_tags_to_moods():
    """Converts tags to moods and stores them in a new database."""
    print("Connecting to the input database...")
    conn = sqlite3.connect(INPUT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT track_id, mood_tags FROM tracks")
    tracks = cursor.fetchall()
    conn.close()
    print(f"Retrieved {len(tracks)} tracks from input database.")

    mood_tracks = []
    for track_id, mood_tags in tracks:
        if not mood_tags:
            print(f"Track {track_id} has no mood tags, skipping...")
            continue
        
        track_moods = set()
        for tag in mood_tags.split(","):
            tag = tag.strip()  # Clean up any extra spaces
            if tag in reversed_tag_map:
                track_moods.update(reversed_tag_map[tag])
            else:
                print(f"Tag '{tag}' not found in reversed_tag_map. Skipping...")
        
        if track_moods:
            mood_tracks.append((track_id, ",".join(sorted(track_moods))))
        else:
            print(f"Track {track_id} has no valid mood mappings, skipping...")

    print(f"Found {len(mood_tracks)} tracks with valid moods.")
    
    # Insert into new DB
    print("Inserting mood tracks into the new database...")
    conn = sqlite3.connect(OUTPUT_DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO tracks (track_id, mood_tags)
        VALUES (?, ?)
    """, mood_tracks)
    conn.commit()
    conn.close()
    
    print(f"Mood database created with {len(mood_tracks)} tracks.")

if __name__ == "__main__":
    create_mood_db()
    convert_tags_to_moods()
