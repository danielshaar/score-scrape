import modal
from pathlib import Path


scrape_image = modal.Image.debian_slim().pip_install(
    "imagehash", "opencv-python-headless", "pillow", "reportlab"
)
app = modal.App("score-scrape")


@app.function()
def score_bounding_box(frame):
    # TODO(sam): Implement.
    return [1250, 0, 1200, 900]


@app.function(image=scrape_image)
def scrape_score(filename, video_bytes):
    import cv2
    import imagehash
    import os
    import tempfile
    from PIL import Image
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.lib.pagesizes import A4
    from io import BytesIO

    def resize_image(img, target_width):
        img_w, img_h = img.size
        if img_w > target_width:
            scale_factor = target_width / img_w
            img_w = target_width
            img_h = int(img_h * scale_factor)
            img = img.resize((int(img_w), img_h), Image.LANCZOS)
        return img

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
        cropped_frame = frames[i][y : y + h, x : x + w]
        cv2.imwrite(cropped_frame_path, cropped_frame)
        image_hash = imagehash.whash(Image.open(cropped_frame_path))

        # TODO(dshaar): See if we can make the threshold tighter. This worked on a diverse score, but it may not on
        # repetitive scores.
        if not previous_image_hash or image_hash - previous_image_hash > 10:
            previous_image_hash = image_hash
            final_frame_paths.append(cropped_frame_path)

        i += 1

    # Open all images
    images = [Image.open(img) for img in final_frame_paths]

    # Create a new PDF
    pdf_filename = "stitched_image.pdf"
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    pdf_w, pdf_h = A4

    # Initialize variables
    page_height_left = pdf_h
    page_number = 1

    for img in images:
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


@app.local_entrypoint()
def main(filename):
    file_path = Path(filename)
    pdf_bytes = scrape_score.remote(file_path.name, file_path.read_bytes())
    output_path = Path("output.pdf")
    output_path.write_bytes(pdf_bytes)
