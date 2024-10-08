import modal
from pathlib import Path

scrape_image = modal.Image.debian_slim().pip_install("opencv-python-headless")

app = modal.App("score-scrape")

@app.function(image=scrape_image)
def scrape_score(filename, video_bytes):
    import cv2

    file_path = Path(f"/videos/{filename}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    while True:
        frame_count += 1

        more_video_left, frame = cap.read()
        if not more_video_left:
            break
        
        # The videos are 30 fps, so this snaps a frame every 2 seconds.
        if frame_count % 60 != 0:
            continue
        
        # Save the frame to a file.
        cv2.imwrite(f"/raw_frames/frame_{frame_count // 60}.jpg", frame)
    
    cap.release()

    return frame_count // 60

@app.local_entrypoint()
def main(filename):
    file_path = Path(filename)
    print(scrape_score.remote(file_path.name, file_path.read_bytes()))