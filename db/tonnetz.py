## Script to convert mp3 files into tonnetz arrays for insertion into NN ## 
## Not finalized or referenced yet ## 

import essentia.standard as es
import numpy as np
import os
import sqlite3
import time

# Paths
FOLDER_PATH = '/volumes/data/mp3_files'  # Folder containing MP3 files
DB_PATH = '/volumes/data/tonnetz.db'  # SQLite database path
TARGET_LENGTH = 1000  # Desired length of Tonnetz arrays

def load_and_extract_tonnetz(file_path, target_length=TARGET_LENGTH):
    start_time = time.time()
    print(f"Processing: {file_path}")
    
    # Load the audio file
    audio = es.MonoLoader(filename=file_path)()
    print(f"Loaded audio, {len(audio)} samples")
    
    # Extract Tonnetz features
    tonnetz = es.Tonnetz()(audio)
    print(f"Extracted Tonnetz shape: {tonnetz.shape}")
    
    # Normalize the Tonnetz array
    tonnetz_normalized = (tonnetz - np.mean(tonnetz)) / np.std(tonnetz)
    
    elapsed_time = time.time() - start_time
    print(f"Feature extraction completed in {elapsed_time:.2f} seconds.")
    
    return tonnetz_normalized

def process_one_mp3(db_path=DB_PATH):
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tonnetz (
            track_id TEXT PRIMARY KEY,
            tonnetz_array BLOB
        )
    ''')
    conn.commit()
    
    # Find one unprocessed MP3 file
    file_paths = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) 
                  if f.endswith('.mp3') and not f.startswith("._")]
    print(f"Found {len(file_paths)} MP3 files.")
    
    for file_path in file_paths:
        track_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check if track ID already exists in DB
        cursor.execute('SELECT track_id FROM tonnetz WHERE track_id = ?', (track_id,))
        if cursor.fetchone() is not None:
            print(f"Track ID {track_id} already exists. Skipping.")
            continue
        
        # Extract Tonnetz array
        tonnetz_array = load_and_extract_tonnetz(file_path)
        
        # Convert array to binary format
        tonnetz_blob = tonnetz_array.tobytes()
        
        # Insert into database
        cursor.execute('''
            INSERT INTO tonnetz (track_id, tonnetz_array)
            VALUES (?, ?)
        ''', (track_id, tonnetz_blob))
        conn.commit()
        print(f"Inserted track: {track_id}")
        break  # Only process one file at a time
    
    conn.close()
    print("Processing complete.")

if __name__ == "__main__":
    process_one_mp3()
