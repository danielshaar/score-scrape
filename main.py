import modal

scrape_image = modal.Image.debian_slim().pip_install("opencv-python-headless")

app = modal.App("score-scrape")

@app.function(
    image=scrape_image,
    mounts=[
        modal.Mount.from_local_file(
            local_path="~/score-scrape/sample-vid.mp4",
            remote_path="/sample-vid.mp4",
        ),
    ],
)
def scrape_score():
    import cv2

    cap = cv2.VideoCapture("/sample-vid.mp4")
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
def main():
    print(scrape_score.remote())