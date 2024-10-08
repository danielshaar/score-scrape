import modal
import os

scrape_image = modal.Image.debian_slim()

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
    return os.stat("/root/sample-vid.mp4").st_size / (1024 * 1024)

@app.local_entrypoint()
def main():
    print(scrape_score.remote())