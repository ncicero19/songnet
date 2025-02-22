**Overview**
- This project is designed to process and analyze audio tracks by extracting Harmonic Pitch Class Profiles (HPCP)
  and managing track metadata in an SQLite database. MP3 files and track mood metadata are downloaded from the Jamendo.com
  free API, ensuring that only applicable data is retained. HPCP features are extracted from the MP3 files using
  the Essentia Python extension and normalized to a fixed frame length to maintain consistency. Song mood metadata is
  then converted into a matrix for validation. Then the final data is inserted into a convolutional neural network. 
  
**Features**
- **Data Assembly:** used the Jamendo Free API to scrape mp3 files and metadata and assemble into DB.
- **SQLite Storage:** metadata and HPCP arrays stored effeciently in SQLite3 DB. 
- **Data refinement:** tags filtered and mapped to overarching themes then eventually convered into 1 x 26 arrays,
  durations filtered, databases automatically deal with discrepancies to match.
- **HPCP Extraction:** Harmonic Pitch Class Profile chromagrams extracted from MP3 files using Essentia Python extension
  and then standardized to one size representing roughly 30 seconds of audio. 
- **Data Testing and Cleanup Scripts:** scripts after each download to test data and correct any errors

**Requirements**
- Python3.13
- Numpy
- Sqlite3 (should be predownloaded for Mac users)
- OS and Time modules
- Python3.11, Essentia and Numpy1.26.4 (installed in a virtual environment to avoid conflicts) (for tonnetz.py)

**Setup** 
1. Install dependencies
2. Start with /scrape/api_final.py (roughly 3-4 hours)
     - Update input variables (SAVE_DIR, DB_PATH, MAX_TRACKS, LIMIT) to your liking
     - I'd recommend starting with a small MAX_TRACKS to make sure it works 
     - Check parameters (full info on parameters can be found at https://developer.jamendo.com/v3.0/tracks)
     - Run script
3. Run /db/data_test.py (instant)
     - Make sure the same number of tracks were downloaded to each place and no tracks under 45 seconds were downloaded
4. Run /db/tonnetz.py (Roughly 1-2 hours)
     - Consult with dependencies.txt first for instructions
     - Change endpoints but leave MAX_FRAMES alone
5. Run /db/convert_tags.py (instant)
     - Follow the instructions carefully, make sure you have already created a new column in your db 
**Usage (Expected Output)** 

**Notes** 
- The hop size and frame size are predefined in hpcp.py.
- The parameters are predefined in final_tracks.py
- You need to consult with dependencies.txt before running tonnetz.py


**License**
 - This project is open source and available under the MIT License. 

**Contact** 
- For any questions, please contact ncicero19@gmail.com
