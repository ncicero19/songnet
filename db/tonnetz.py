## Pipeline to create HPCP chromagrams from stored mp3_files ##
## Skips tracks that are too short (not enough frames) and duplicates##
## Uses hpcp.py to convert, the rest is just a pipeline ## 
## SEE DEPENDENCIES.TXT FOR THE REQUIREMENTS TO RUN THIS SCRIPT ##

import essentia.standard as es
import numpy as np
import os
import sqlite3
import time
import json
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
    hpcp_array = hpcp_array[start:end, :]
    
    # Normalize the HPCP array
    hpcp_normalized = (hpcp_array - np.mean(hpcp_array)) / np.std(hpcp_array)
    
    elapsed_time = time.time() - start_time
    return hpcp_normalized

def process_mp3_files(db_path=DB_PATH):
    # Connect to SQLite database
    counter = 0
    skip_counter = 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ensure the table exists with JSON storage
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hpcp (
            track_id TEXT PRIMARY KEY,
            hpcp_array TEXT
        )
    ''')
    conn.commit()
    
    # Find MP3 files
    file_paths = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) 
                  if f.endswith('.mp3') and not f.startswith("._")]
    total_files = len(file_paths)
    print(f"Found {total_files} files")
    
    # Check how many are already in the database for the printed download update denominator
    existing_files_count = 0
    for file_path in file_paths:
        track_id = os.path.splitext(os.path.basename(file_path))[0]
        cursor.execute('SELECT hpcp_array FROM hpcp WHERE track_id = ?', (track_id,))
        row = cursor.fetchone()
    
        if row is not None and row[0]:  # Check if data exists and is not empty
            try:
                existing_array = np.array(json.loads(row[0]))
                if existing_array.shape == (TARGET_LENGTH, 12):  # Correct shape
                    existing_files_count += 1
            except (json.JSONDecodeError, ValueError):
                pass  # Corrupt data will be updated, so it's not counted as existing

    # Adjust total_files to reflect only new or incorrectly stored files
    total_files -= existing_files_count

    for file_path in file_paths:
        track_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check if track ID exists and validate shape
        cursor.execute('SELECT hpcp_array FROM hpcp WHERE track_id = ?', (track_id,))
        row = cursor.fetchone()
        
        if row is not None and row[0]:
            try:
                existing_array = np.array(json.loads(row[0]))
                if existing_array.shape == (TARGET_LENGTH, 12):
                    print(f"{track_id} already exists in the correct format. ")
                    continue
                else:
                    print(f"Track ID {track_id} has an incorrectly shaped array. Updating.")
            except (json.JSONDecodeError, ValueError):
                print(f"Track ID {track_id} has corrupt data. Updating.")
        
        try:
            # Extract HPCP array
            hpcp_array = load_and_extract_hpcp(file_path)
            
            # Convert array to JSON format
            hpcp_json = json.dumps(hpcp_array.tolist())
            
            if row is None:
                # Insert new entry
                cursor.execute('''
                    INSERT INTO hpcp (track_id, hpcp_array)
                    VALUES (?, ?)
                ''', (track_id, hpcp_json))
            else:
                # Update existing entry
                cursor.execute('''
                    UPDATE hpcp SET hpcp_array = ? WHERE track_id = ?
                ''', (hpcp_json, track_id))
            
            conn.commit()
            counter += 1
            print(f"{track_id} processed. {counter}/{total_files}")
        
        except ValueError as e:
            skip_counter += 1
            print(f"Skipping {track_id}: {e}")
    
    conn.close()
    print(f"Processing complete. {counter} arrays processed. Skipped {skip_counter} tracks of invalid length.")

if __name__ == "__main__":
    process_mp3_files()
