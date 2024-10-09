import modal
from pathlib import Path


app = modal.App("score-scrape")


cuda_version = "12.4.0"
os_version = "ubuntu22.04"
segment_image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_version}-devel-{os_version}", add_python="3.10")
    .apt_install("clang", "git", "libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless", "pillow", "poetry", "torch", "torchvision", "wheel")
    .pip_install("git+https://github.com/luca-medeiros/lang-segment-anything.git", gpu="A100")
)
with segment_image.imports():
    from PIL import Image


@app.cls(keep_warm=1, concurrency_limit=20, gpu="A100", image=segment_image, volumes={"/mods": modal.Volume.from_name("mods")})
class Model:
    @modal.enter()
    def load_model(self):
        print("importing")
        from PIL import Image
        from lang_sam import LangSAM
        print("imported")
        self.model = LangSAM(sam_type="vit_h", ckpt_path="/mods/sam_vit_h_4b8939.pth")
        print("constructed model")

    @modal.method()
    def predict(self, raw_bytes):
        from PIL import Image
        import cv2
        cv2.imwrite("/root/subject_file.png", raw_bytes)
        image_pil = Image.open("/root/subject_file.png").convert("RGB")
        text_prompt = "sheet music score"
        print("generating masks")

        masks, boxes, phrases, logits = self.model.predict(image_pil, text_prompt)
        stacked = zip(masks, boxes, phrases, logits)
        stacked = sorted(stacked, key=lambda x: x[3], reverse=True)
        masks, boxes, phrases, logits = zip(*stacked)

        print("masks generated")
        print(logits)
        #print(masks)
        print(boxes)
        print(phrases)
        box = [round(x) for x in boxes[0].tolist()]
        return {"rect": box, "logit": float(logits[0])}


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
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox["rect"]
        score_segment_image = Image.fromarray(frames[i][y : y + h, x : x + w])
        trimmed_score_image = remove_border(score_segment_image)
        if trimmed_score_image:
            image_hash = imagehash.whash(trimmed_score_image)

            # TODO(dshaar): See if we can make the threshold tighter. This worked on a diverse score, but it may not on
            # repetitive scores.
            if not previous_image_hash or image_hash - previous_image_hash > 5:
                previous_image_hash = image_hash
                unique_score_images.append(trimmed_score_image)

    return unique_score_images


def resize_image(img, target_width):
        img_w, img_h = img.size
        if img_w > target_width:
            scale_factor = target_width / img_w
            img_w = target_width
            img_h = int(img_h * scale_factor)
            img = img.resize((int(img_w), img_h), Image.LANCZOS)
        return img


def stitch_pdf(unique_score_images):
    import os
    import tempfile
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.lib.pagesizes import A4
    from io import BytesIO

    # Create a new PDF
    pdf_filename = "stitched_image.pdf"
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    pdf_w, pdf_h = A4

    # Initialize variables
    page_height_left = pdf_h
    page_number = 1

    for img in unique_score_images:
        # Resize the image to the width of the PDF page
        img = resize_image(img, pdf_w)
        img_w, img_h = img.size

        # Check if there's enough space on the current page
        if page_height_left - img_h < 0:
            # If not enough space, start a new page and reset the page_height_left
            c.showPage()
            page_height_left = pdf_h
            page_number += 1

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            img.save(temp_file.name, format="PNG")

            # Draw the image on the PDF on the current page
            c.drawImage(
                temp_file.name, 0, page_height_left - img_h, width=img_w, height=img_h
            )

        # Remove the temporary file
        os.unlink(temp_file.name)

        # Update page_height_left for the next image
        page_height_left -= img_h

    # Save the PDF
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
    model = Model()
    bboxes = model.predict.map(frames)
    unique_score_images = crop_and_dedupe_frames(frames, bboxes)
    return stitch_pdf(unique_score_images)


@app.local_entrypoint()
def main(filename):
    file_path = Path(filename)
    pdf_bytes = scrape_score.remote(file_path.name, file_path.read_bytes())
    Path("output.pdf").write_bytes(pdf_bytes)
