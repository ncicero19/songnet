## Converts all tags into an array to be used for validation data and stores that as a new column in the tag db## 
## The array has an entry for each of the approved tags, and a 1 signifies having that tag ##
## This assumes you have already created a new column in your tag db -- if not, this can be run simply##

import sqlite3
import numpy as np
from tag_map import tag_map  # Assuming tag_map.py contains the dictionary

# Database connection
db_path = "/volumes/data/final_tracks.db"  
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch song IDs and their associated tags
cursor.execute("SELECT track_id, mood_tags FROM tracks")  
rows = cursor.fetchall()

# Organize data by track
songs = {}
for track_id, mood_tags in rows:
    if track_id not in songs:
        songs[track_id] = []
    songs[track_id].extend(mood_tags.split(','))  # Assuming tags are stored as comma-separated strings

def create_matrix(input_tags):
    """Create a binary matrix where each row corresponds to a tag in tag_map."""
    matrix = np.zeros((len(tag_map), 1), dtype=int)

    for i, tag in enumerate(tag_map.keys()):
        if tag in input_tags:
            matrix[i][0] = 1  # Set row to 1 if the tag is present

    return matrix

# Generate and update matrices for each song
for track_id, tag_list in songs.items():
    matrix = create_matrix(tag_list)
    
    # Convert matrix to a comma-separated string for direct storage
    tag_array_str = ",".join(map(str, matrix.flatten()))
    print(tag_array_str)

    # Update only if the column already exists
    cursor.execute("""
        UPDATE tracks
        SET tag_array = ?
        WHERE track_id = ?;
    """, (tag_array_str, track_id))

# Commit changes and close connection
conn.commit()
conn.close()

print("Tag arrays successfully updated in the database.")

