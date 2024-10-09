import modal
from pathlib import Path


scrape_image = modal.Image.debian_slim().pip_install(
    "imagehash", "opencv-python-headless", "pillow", "reportlab"
)
app = modal.App("score-scrape")


@app.function(image=scrape_image)
def scrape_score(filename, video_bytes):
    import cv2
    import imagehash
    import numpy as np
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
    model = Model()

    seg_results = model.predict_batch.remote(frames)

    for i, res in enumerate(seg_results):
        x, y, w, h = res["rect"]
        print("res", res)
        cropped_frame_path = Path(f"/cropped_frames/frame_{i}.png")
        cropped_frame_path.parent.mkdir(parents=True, exist_ok=True)
        cropped_frame = frames[i][y : y + h, x : x + w]
        cropped_frame2 = trim(Image.fromarray(cropped_frame))
        if cropped_frame2:
            cropped_frame = cropped_frame2

        cropped_frame = np.array(cropped_frame)
        cv2.imwrite(cropped_frame_path, cropped_frame)
        image_hash = imagehash.whash(Image.open(cropped_frame_path))

        # TODO(dshaar): See if we can make the threshold tighter. This worked on a diverse score, but it may not on
        # repetitive scores.
        if not previous_image_hash or image_hash - previous_image_hash > 10:
            previous_image_hash = image_hash
            final_frame_paths.append(cropped_frame_path)


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
    # for debugging
    for res in seg_results:
        print(res)
    return pdf_bytes, images


@app.local_entrypoint()
def main(filename):
    from PIL import Image
    file_path = Path(filename)
    pdf_bytes, images = scrape_score.remote(file_path.name, file_path.read_bytes())
    output_path = Path("output.pdf")
    output_path.write_bytes(pdf_bytes)
    for k, image in enumerate(images):
        new_path = Path(f"frames/frame_{k}.png")
        image.save(new_path)
        


seg_image = (
    modal.Image.debian_slim()
    .apt_install('git')
    .apt_install('wget')    
    .apt_install(['libglib2.0-0', 'libsm6', 'libxrender1', 'libxext6', 'libgl1'])
    .run_commands([
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get -y install cuda-toolkit-12-4",
        ])
    .env({'CUDA_HOME': '/usr/local/cuda-12.4'})
    .pip_install("torch", "torchvision", "opencv-python-headless") 
    .run_commands([
        "echo $CUDA_HOME",
        "ls -l $CUDA_HOME",
        "cd /root && git clone https://github.com/luca-medeiros/lang-segment-anything",
        "cd /root/lang-segment-anything && pip install -e .",
    ], gpu="A100")
)

@app.cls(keep_warm=1, gpu="A100", image=seg_image)
class Model:
    @modal.enter()
    def load_model(self):
        print("importing")
        from PIL import Image
        from lang_sam import LangSAM
        print("imported")
        self.model = LangSAM()
        print("constructed model")

    def _predict(self, raw_bytes):
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
    
    @modal.method()
    def predict(self, image):
        return self._predict(image)

    @modal.method()
    def predict_batch(self, image_list):
        return [
            self._predict(image) for image in image_list
        ]


def trim(im):
    from PIL import Image, ImageChops
    bg = Image.new(im.mode, im.size)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    
