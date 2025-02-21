## Pipeline to create HPCP chromagrams from stored mp3_files ##
## Skips tracks that are too short (not enough frames) and duplicates##
## Uses hpcp.py to convert, the rest is just a pipeline ## 
## SEE DEPENDENCIES.TXT FOR THE REQUIREMENTS TO RUN THIS SCRIPT ##

import essentia.standard as es
import numpy as np
import os
import sqlite3
import time
from hpcp import hpcp

# Paths
FOLDER_PATH = '/volumes/data/mp3_files'  # Folder containing MP3 files
DB_PATH = '/volumes/data/tonnetz.db'  # SQLite database path
TARGET_LENGTH = 2500  # Desired length of HPCP arrays

def load_and_extract_hpcp(file_path, target_length=TARGET_LENGTH):
    start_time = time.time()
    
    # Extract HPCP features
    hpcp_array = hpcp(file_path)
    print(f"Extracted HPCP shape: {hpcp_array.shape}")
    
    # Normalize length to correspond to ~30 seconds from the middle
    current_length = hpcp_array.shape[0]
    required_frames = target_length
    
    if current_length < required_frames:
        raise ValueError(f"Track {file_path} is too short. Only {current_length} frames available, but {required_frames} required.")
    
    start = max(0, (current_length - required_frames) // 2)
    end = start + required_frames
    hpcp_array = hpcp_array[:, start:end]
    
    # Normalize the HPCP array
    hpcp_normalized = (hpcp_array - np.mean(hpcp_array)) / np.std(hpcp_array)
    
    elapsed_time = time.time() - start_time
    # print(f"Feature extraction completed in {elapsed_time:.2f} seconds.")
    
    return hpcp_normalized


def process_one_mp3(db_path=DB_PATH):
    # Connect to SQLite database
    counter = 0
    skip_counter = 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hpcp (
            track_id TEXT PRIMARY KEY,
            hpcp_array BLOB
        )
    ''')
    conn.commit()
    
    # Find one unprocessed MP3 file
    file_paths = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) 
                  if f.endswith('.mp3') and not f.startswith("._")]
    total_files = len(file_paths)
    print(f"Found {total_files} files")
    
    
    for file_path in file_paths:
        track_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check if track ID already exists in DB
        cursor.execute('SELECT track_id FROM hpcp WHERE track_id = ?', (track_id,))
        if cursor.fetchone() is not None:
            print(f"Track ID {track_id} already exists. Skipping.")
            continue
        
        try:
            # Extract HPCP array
            hpcp_array = load_and_extract_hpcp(file_path)
            
            # Convert array to binary format
            hpcp_blob = hpcp_array.tobytes()
            
            # Insert into database
            cursor.execute('''
                INSERT INTO hpcp (track_id, hpcp_array)
                VALUES (?, ?)
            ''', (track_id, hpcp_blob))
            conn.commit()
            counter += 1
            print(f"{track_id} inserted. {counter}/{total_files}")
        
        except ValueError as e:
            skip_counter += 1
            print(f"Skipping {track_id}: {e}")
    
    conn.close()
    print(f"Processing complete. {counter} arrays downloaded. Skipped {skip_counter} tracks of invalid length.")


if __name__ == "__main__":
    process_one_mp3()
