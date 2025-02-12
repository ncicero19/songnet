## Final db cleanup, synchronizes DBs and also filters out tracks less than 46 seconds (1024 frames) ##

import sqlite3
import os

# Paths
FINAL_TRACKS_DB = "/volumes/data/final_tracks.db"
DURATION_DB = "/volumes/data/durations.db"
MP3_DIR = "/volumes/data/mp3_files"

# Connect to databases
def get_valid_tracks():
    """Returns a set of track_ids that exist in all three sources."""
    conn1 = sqlite3.connect(FINAL_TRACKS_DB)
    conn2 = sqlite3.connect(DURATION_DB)
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()
    
    cursor1.execute("SELECT track_id FROM tracks")
    final_tracks = {row[0] for row in cursor1.fetchall()}
    
    cursor2.execute("SELECT track_id, duration FROM durations")
    durations = {row[0]: row[1] for row in cursor2.fetchall()}
    
    mp3_files = {f.replace(".mp3", "") for f in os.listdir(MP3_DIR) if f.endswith(".mp3") and not f.startswith("._")}
    
    conn1.close()
    conn2.close()
    
    valid_tracks = final_tracks & durations.keys() & mp3_files
    
    # Filter out tracks with duration < 46 seconds
    valid_tracks = {tid for tid in valid_tracks if durations[tid] >= 46}
    
    return valid_tracks, durations

def cleanup_invalid_tracks():
    """Deletes tracks that do not exist in all three sources or have duration < 46 seconds."""
    valid_tracks, durations = get_valid_tracks()
    
    conn1 = sqlite3.connect(FINAL_TRACKS_DB)
    conn2 = sqlite3.connect(DURATION_DB)
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()
    
    # Delete from final_tracks.db
    cursor1.execute(f"DELETE FROM tracks WHERE track_id NOT IN ({','.join('?' * len(valid_tracks))})", tuple(valid_tracks))
    conn1.commit()
    
    # Delete from durations.db
    cursor2.execute(f"DELETE FROM durations WHERE track_id NOT IN ({','.join('?' * len(valid_tracks))})", tuple(valid_tracks))
    conn2.commit()
    
    conn1.close()
    conn2.close()
    
    # Delete invalid MP3 files
    for file in os.listdir(MP3_DIR):
        if file.endswith(".mp3") and not file.startswith("._"):
            track_id = file.replace(".mp3", "")
            if track_id not in valid_tracks:
                os.remove(os.path.join(MP3_DIR, file))
                print(f"Deleted MP3 file: {file}")
    
    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup_invalid_tracks()

