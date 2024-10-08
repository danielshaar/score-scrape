import modal
import cv2

scrape_image = modal.Image.debian_slim().pip_install("opencv-python")

app = modal.App("score-scrape")

@app.function(
    image=scrape_image,
    mounts=[
        modal.Mount.from_local_file(
            local_path="~/score-scrape/sample-vid.mp4",
            remote_path="/root/sample-vid.mp4",
        ),
    ],
)
def scrape_score():
    cap = cv2.VideoCapture("/root/sample-vid.mp4")
    frame_count = 0
    while True:
        frame_count += 1
        video_over, frame = cap.read()
        if video_over:
            break
        
        # The videos are 30 fps, so this snaps a frame every 2 seconds.
        if frame_count % 60 != 0:
            continue
        
        # Save the frame to a file.
        cv2.imwrite(f"/root/raw_frames/frame_{frame_count // 60}.jpg", frame)
    
    cap.release()

@app.local_entrypoint()
def main():
    print(scrape_score.remote())