import modal
from pathlib import Path

scrape_image = modal.Image.debian_slim().pip_install("imagehash", "opencv-python-headless", "pillow")
app = modal.App("score-scrape")


@app.function()
def score_bounding_box(frame):
    # TODO(sam): Implement.
    return [1250, 0, 1200, 900]


@app.function(image=scrape_image)
def scrape_score(filename, video_bytes):
    import cv2
    import imagehash
    from PIL import Image

    file_path = Path(f"/videos/{filename}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    frames = []
    while True:
        frame_count += 1

        more_video_left, frame = cap.read()
        if not more_video_left:
            break
        
        # The videos are 30 fps, so this snaps a frame every 2 seconds.
        if frame_count % 60 != 0:
            continue

        frames.append(frame)
    
    cap.release()

    # Crop and dedupe frames.
    i = 0
    previous_image_hash = None
    final_frame_paths = []
    for x, y, w, h in score_bounding_box.map(frames):
        cropped_frame_path = Path(f"/cropped_frames/frame_{i}.png")
        cropped_frame_path.parent.mkdir(parents=True, exist_ok=True)
        cropped_frame = frames[i][y:y+h, x:x+w]
        cv2.imwrite(cropped_frame_path, cropped_frame)
        image_hash = imagehash.whash(Image.open(cropped_frame_path))

        # TODO(dshaar): See if we can make the threshold tighter. This worked on a diverse score, but it may not on
        # repetitive scores.
        if not previous_image_hash or image_hash - previous_image_hash > 10:
            previous_image_hash = image_hash
            final_frame_paths.append(cropped_frame_path)
        
        i += 1
    
    # TODO(anna): Stitch frames into PDF.
    return len(final_frame_paths)

@app.local_entrypoint()
def main(filename):
    file_path = Path(filename)
    print(scrape_score.remote(file_path.name, file_path.read_bytes()))