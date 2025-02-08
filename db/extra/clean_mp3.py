## Clean up flash drive storage by deleting all mp3 files of songs with unapplicable tags (tags not in final_tracks.db) ##

import sqlite3
import os

# Paths
DB_PATH = "/volumes/data/final_tracks.db"
MP3_DIR = "/volumes/data/mp3_files"

# Connect to the database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get table name
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]

if not tables:
    print("Error: No tables found in the database.")
    conn.close()
    exit(1)

# Print all available tables for debugging
print("Tables found in database:", tables)

# Identify the correct table
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
existing_tables = [row[0] for row in cursor.fetchall()]

if not existing_tables:
    print("Error: No valid tables found in the database.")
    conn.close()
    exit(1)

# Use the first non-system table found
table_name = existing_tables[0]
print(f"Using table: {table_name}")

# Get all track IDs from the database
cursor.execute(f"SELECT track_id FROM {table_name}")
valid_track_ids = set(str(row[0]) for row in cursor.fetchall())
conn.close()

# Iterate through mp3 files and delete unlisted ones
for filename in os.listdir(MP3_DIR):
    if filename.endswith(".mp3"):
        track_id = os.path.splitext(filename)[0]  # Assuming filename is "track_id.mp3"
        if track_id not in valid_track_ids:
            file_path = os.path.join(MP3_DIR, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

print("Cleanup complete.")
