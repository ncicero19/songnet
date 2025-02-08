##Presents all tags one by one with a y/n option to keep or throw out##
## Will become obselete once the DB is finished. Used for creation##
import sqlite3

db_path = "/volumes/data/tracks.db"
out_file = "selected_tags.txt"

def get_mood_tags():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT mood_tags FROM tracks")
    tags = cursor.fetchall()
    conn.close()
    unique_tags = set()
    for tag_tuple in tags:
        if tag_tuple[0]:
            unique_tags.update(tag_tuple[0].split(","))
    return sorted(unique_tags)

def filter_tags(tags):
    selected_tags = []
    for tag in tags:
        choice = input(f"Include tag '{tag}'? (y/n): ").strip().lower()
        if choice == 'y':
            selected_tags.append(tag)
    
    with open(out_file, "w") as f:
        for tag in selected_tags:
            f.write(tag + "\n")
    print("Tag filtering complete. Selected tags saved to selected_tags.txt")

tags = get_mood_tags()
filter_tags(tags)
