import modal
from pathlib import Path


app = modal.App("score-scrape")


cuda_version = "12.4.0"
os_version = "ubuntu22.04"
segment_image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_version}-devel-{os_version}", add_python="3.10")
    .apt_install("clang", "git", "libgl1", "libglib2.0-0")
    .pip_install("pillow", "poetry", "torch", "torchvision", "wheel")
    .pip_install("git+https://github.com/luca-medeiros/lang-segment-anything.git", gpu="A10G")
)
with segment_image.imports():
    from PIL import Image


@app.cls(
    keep_warm=1,
    concurrency_limit=2,
    gpu="A10G",
    image=segment_image,
    volumes={"/mods": modal.Volume.from_name("mods")}
)
class ExtractScoreModel:
    @modal.enter()
    def load_model(self):
        from lang_sam import LangSAM

        self.model = LangSAM(sam_type="vit_h", ckpt_path="/mods/sam_vit_h_4b8939.pth")

    @modal.method()
    def predict(self, frame):
        bboxes = self.model.predict(Image.fromarray(frame).convert("RGB"), "sheet music score")[1]
        largest_bbox = sorted(bboxes, key=lambda bbox: bbox[2] * bbox[3], reverse=True)[0]
        return [round(x) for x in largest_bbox.tolist()]


def get_frames(file_path):
    import cv2

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
    return frames


def remove_border(score_segment_image):
    from PIL import ImageChops

    # Find the bounding box of the non-black pixels.
    blank_background = Image.new(score_segment_image.mode, score_segment_image.size)
    diff = ImageChops.difference(score_segment_image, blank_background)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    # If we found a bounding box, tighten it up by one pixel to remove the border.
    if bbox:
        no_border_bbox = (bbox[0] + 1, bbox[1] + 1, bbox[2] - 2, bbox[3] - 2)
        return score_segment_image.crop(no_border_bbox)


def crop_and_dedupe_frames(frames, bboxes):
    import imagehash

    previous_image_hash = None
    unique_score_images = []
    for i, (x, y, w, h) in enumerate(bboxes):
        score_segment_image = Image.fromarray(frames[i][y : y + h, x : x + w])
        trimmed_score_image = remove_border(score_segment_image)
        if trimmed_score_image:
            image_hash = imagehash.whash(trimmed_score_image)

            if not previous_image_hash or image_hash - previous_image_hash > 5:
                previous_image_hash = image_hash
                unique_score_images.append(trimmed_score_image)

    return unique_score_images


def scale_to_page(score_image, page_w):
    w, h = score_image.size
    new_w = 0.95 * page_w
    new_h = h * new_w / w
    return score_image.resize((int(new_w), int(new_h)), Image.LANCZOS)


def draw_page(c, page_images, page_h_left):
    from reportlab.lib.pagesizes import A4

    page_w, page_h = A4

    h_offset = page_h - 0.5 * page_h_left
    for page_image in page_images:
        w, h = page_image.size
        h_offset -= h
        c.drawInlineImage(page_image, 0.025 * page_w, h_offset, width=w, height=h)


def stitch_pdf(unique_score_images):
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    page_w, page_h = A4

    page_h_left = page_h
    page_images = []
    for score_image in unique_score_images:
        score_image = scale_to_page(score_image, page_w)
        w, h = score_image.size

        # If we can't fit another image on the page, draw the page.
        if page_h_left - h < -10:
            draw_page(c, page_images, page_h_left)
            c.showPage()
            page_h_left = page_h
            page_images = []
        
        page_h_left -= h
        page_images.append(score_image)
    
    # Draw the last page.
    if page_images:
        draw_page(c, page_images, page_h_left)

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


scrape_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("imagehash", "opencv-python-headless", "pillow", "reportlab")
)


@app.function(image=scrape_image)
def scrape_score(filename, video_bytes):
    file_path = Path(f"/videos/{filename}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(video_bytes)

    frames = get_frames(file_path)
    extract_score_model = ExtractScoreModel()
    bboxes = extract_score_model.predict.map(frames)
    unique_score_images = crop_and_dedupe_frames(frames, bboxes)
    return stitch_pdf(unique_score_images)


@app.local_entrypoint()
def main(filename):
    file_path = Path(filename)
    pdf_bytes = scrape_score.remote(file_path.name, file_path.read_bytes())
    Path("output.pdf").write_bytes(pdf_bytes)
