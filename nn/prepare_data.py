## Prepares data by turning it into .pt tensors and adding a dimension for color ## 
## The chromagraph dataset is 1 x 12 x 2500 x num_tracks and the key is 26 x 1 ##

import sqlite3
import torch
import json
import numpy as np

# Database paths
CHROMA_DB = "/volumes/data/tonnetz.db"
TAGS_DB = "/volumes/data/final_tracks.db"
OUTPUT_FILE = "/volumes/data/final_data.pt"

def load_data():
    """Loads and aligns chromagram and tag data from SQLite databases."""
    
    # Connect to databases
    chroma_conn = sqlite3.connect(CHROMA_DB)
    tags_conn = sqlite3.connect(TAGS_DB)
    chroma_cursor = chroma_conn.cursor()
    tags_cursor = tags_conn.cursor()
    
    # Fetch track_ids from both databases
    chroma_cursor.execute("SELECT track_id FROM hpcp")
    chroma_tracks = set(row[0] for row in chroma_cursor.fetchall())
    
    tags_cursor.execute("SELECT track_id FROM tracks")
    tag_tracks = set(row[0] for row in tags_cursor.fetchall())
    
    # Find common track_ids
    common_tracks = sorted(chroma_tracks & tag_tracks)
    
    chroma_data, tag_data = [], []

    for track_id in common_tracks:
        try:
            # Load chromagram data
            chroma_cursor.execute("SELECT hpcp_array FROM hpcp WHERE track_id = ?", (track_id,))
            chroma_json = chroma_cursor.fetchone()
            if not chroma_json or not chroma_json[0]:
                print(f"Skipping {track_id}: Missing chroma data")
                continue

            chroma_array = np.array(json.loads(chroma_json[0]), dtype=np.float32)

            if chroma_array.shape != (2500, 12):
                print(f"Skipping {track_id}: Unexpected chroma shape {chroma_array.shape}")
                continue
            
            # Load tag data
            tags_cursor.execute("SELECT tag_array FROM tracks WHERE track_id = ?", (track_id,))
            tag_string = tags_cursor.fetchone()
            if not tag_string or not tag_string[0]:
                print(f"Skipping {track_id}: Missing tag data")
                continue

            tag_array = np.array([float(x) for x in tag_string[0].split(",")], dtype=np.float32).reshape(26, 1)
            
            # Store tensors
            chroma_data.append(torch.tensor(chroma_array, dtype=torch.float32))
            tag_data.append(torch.tensor(tag_array, dtype=torch.float32))
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Skipping {track_id}: Data error - {e}")

    if chroma_data and tag_data:
        # Stack into 3D tensors
        chroma_tensor = torch.stack(chroma_data)
        tag_tensor = torch.stack(tag_data)

        # Save as .pt file
        torch.save({"chromagrams": chroma_tensor, "tags": tag_tensor}, OUTPUT_FILE)

        print(f"✅ Processed {len(chroma_data)} tracks and saved to {OUTPUT_FILE}")
    else:
        print("⚠️ No valid data to save.")

    # Close connections
    chroma_conn.close()
    tags_conn.close()

if __name__ == "__main__":
    load_data()
