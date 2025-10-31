import os
import time

CHECK_FOLDER = "save"

MAX_FILES = 3

INTERVAL = 10

def get_pt_files(folder):
    """Return list of .pt files with full path and modification time."""
    files = [
        (os.path.join(folder, f), os.path.getmtime(os.path.join(folder, f)))
        for f in os.listdir(folder)
        if f.endswith(".pt") and os.path.isfile(os.path.join(folder, f))
    ]
    return files

def cleanup_folder(folder, max_files):
    files = get_pt_files(folder)
    if len(files) > max_files:
        files.sort(key=lambda x: x[1])
        num_to_delete = len(files) - max_files
        for i in range(num_to_delete):
            try:
                os.remove(files[i][0])
                print(f"Deleted old checkpoint: {files[i][0]}")
            except Exception as e:
                print(f"Error deleting {files[i][0]}: {e}")

if __name__ == "__main__":
    print(f"Monitoring folder: {CHECK_FOLDER}")
    while True:
        cleanup_folder(CHECK_FOLDER, MAX_FILES)
        time.sleep(INTERVAL)
