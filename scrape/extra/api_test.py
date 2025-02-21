## Test of API output for debugging. ##
## Filters and prints the data for 5 tracks ##


import requests
import api_certs
import alltags
import tag_map

API_URL = "https://api.jamendo.com/v3.0/tracks/"
API_KEY = api_certs.client_id
TAG_LIST = alltags.alltags

def fetch_tracks():
    params = {
        "client_id": API_KEY,
        "format": "json",
        "limit": 5,  # Adjust number of tracks
        "include": "musicinfo",
        "order": "relevance",
        "fuzzytags": ",".join(list(TAG_LIST)[:3]),
        "boost": "popularity_total",
        "audioformat": "mp32"
    }
    
    response = requests.get(API_URL, params=params)

    if response.status_code != 200:
        print(f"Error: {response.text}")  # Print error message if request fails
        return []
    
    data = response.json()
    
    # Extract only relevant fields
    filtered_tracks = []
    for track in data.get("results", []):
        track_info = {
            "track_id": track.get("id"),
            "title": track.get("name"),
            "artist": track.get("artist_name"),
            "mood_tags": track.get("musicinfo", {}).get("tags", {}).get("vartags", []),
            "genre_tags": track.get("musicinfo", {}).get("tags", {}).get("genres", []),
            "audio_link": track.get("audio"),
            "audio_download": track.get("audiodownload"),
        }
        filtered_tracks.append(track_info)

    return filtered_tracks

# Run the function and print results
tracks = fetch_tracks()
for track in tracks:
    print(track)


