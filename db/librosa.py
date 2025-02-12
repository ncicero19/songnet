## Script to convert mp3 files into tonnetz arrays for insertion into NN ## 
## Not finalized or referenced yet ## 

import librosa
import numpy as np
import os
import sqlite3

def load_and_extract_tonnetz(file_path, target_length=1000):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract Tonnetz features with frame size and hop length specified 
    n_fft = 1024
    hop_length = 512
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # Get the middle segment of the Tonnetz array
    current_length = tonnetz.shape[1]
    start = max(0, (current_length - target_length) // 2)
    end = start + target_length
    
    # If the audio is shorter than the target length, pad with zeros
    if current_length < target_length:
        padding = target_length - current_length
        tonnetz = np.pad(tonnetz, ((0, 0), (0, padding)), mode='constant')
    else:
        tonnetz = tonnetz[:, start:end]
    
    # Normalize the Tonnetz array
    tonnetz_normalized = (tonnetz - np.mean(tonnetz)) / np.std(tonnetz)
    
    return tonnetz_normalized

def process_mp3_files(folder_path, target_length=1000, db_path='tonnetz_database.db'):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a table to store Tonnetz arrays and track IDs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tonnetz_data (
            track_id TEXT PRIMARY KEY,
            tonnetz_array BLOB
        )
    ''')
    conn.commit()
    
    # Get all MP3 files in the folder
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp3') and not f.startswith("._")]
    
    for file_path in file_paths:
        # Extract the track ID (file name without extension)
        track_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check if the track ID already exists in the database
        cursor.execute('SELECT track_id FROM tonnetz_data WHERE track_id = ?', (track_id,))
        if cursor.fetchone() is not None:
            print(f"Track ID {track_id} already exists in the database. Skipping.")
            continue
        
        # Extract Tonnetz array
        tonnetz_array = load_and_extract_tonnetz(file_path, target_length)
        
        # Convert the Tonnetz array to a binary format for storage
        tonnetz_blob = tonnetz_array.tobytes()
        
        # Insert the track ID and Tonnetz array into the database
        cursor.execute('''
            INSERT INTO tonnetz_data (track_id, tonnetz_array)
            VALUES (?, ?)
        ''', (track_id, tonnetz_blob))
        conn.commit()
        print(f"Inserted Tonnetz array for track ID: {track_id}")
    
    # Close the database connection
    conn.close()

# Usage
folder_path = '/volumes/data/mp3_files'  # Folder containing MP3 files
target_length = 1000  # Desired length of the Tonnetz arrays
db_path = '/volumes/data/tonnetz.db'  # Path to the SQLite database
process_mp3_files(folder_path, target_length, db_path)