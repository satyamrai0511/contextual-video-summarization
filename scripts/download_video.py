import yt_dlp
import os
import shutil

SAVE_DIR = "dataset/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

videos = {
    "4fOWQfzWHbc": "https://youtu.be/4fOWQfzWHbc",
    "PsFlp3u74mI": "https://youtu.be/PsFlp3u74mI",
    "-k-bg0q3NNw": "https://youtu.be/-k-bg0q3NNw",
    "C4QMPhiF_as": "https://youtu.be/C4QMPhiF_as",
    "eMiPKpXbm9A": "https://youtu.be/eMiPKpXbm9A",
    "ADAmTm_45uk": "https://youtu.be/ADAmTm_45uk",
    "utTlli1Ett0": "https://youtu.be/utTlli1Ett0",
    "p0Nag5w42ys": "https://youtu.be/p0Nag5w42ys",
    "jc0tTOhRVe0": "https://youtu.be/jc0tTOhRVe0",
    "cIu8Xo5yHlI": "https://youtu.be/cIu8Xo5yHlI"
}

# Temporary download format to figure out actual extension
ydl_opts = {
    'outtmpl': os.path.join(SAVE_DIR, '%(id)s.%(ext)s'),
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
    'merge_output_format': 'mp4',
    'quiet': False
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for vid_id, url in videos.items():
        print(f"Downloading {vid_id}...")
        info = ydl.extract_info(url, download=True)
        downloaded_filename = os.path.join(SAVE_DIR, f"{info['id']}.{info['ext']}")
        correct_filename = os.path.join(SAVE_DIR, f"{vid_id}.mp4")

        # Rename file to match what our scripts expect
        if downloaded_filename != correct_filename:
            shutil.move(downloaded_filename, correct_filename)
            print(f"Renamed to {correct_filename}")
        else:
            print(f"Saved as {correct_filename}")
