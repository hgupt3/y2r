import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process first frame in folders.")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the recordings directory')
    parser.add_argument('--text_prompt', type=str, required=True, help='Text prompt for object detection')
    return parser.parse_args()

def process_first_frame_in_folders(
    recordings_dir,
    text_prompt,
    device="cuda",
    sam2_checkpoint="sam2/checkpoints/sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    grounding_model_id="IDEA-Research/grounding-dino-base",
    box_threshold=0.4,
    text_threshold=0.4
):
    """
    1) Loads Grounding DINO + SAM2 models once.
    2) Loops over every subfolder in `recordings_dir`.
    3) If `rgb/0.png` exists there, runs text-based object detection + segmentation on just that frame.
    4) Saves a single black-and-white mask image in a new `masks/0.png` file.
       - White = union of all detected objects
       - Black = background
    """
    # ------------------------------------------------
    # Step A: Build and load models (ONE TIME)
    # ------------------------------------------------
 
    # 1. Build SAM2 model and predictor
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint).to(device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # 2. Build Grounding DINO model and processor
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)

    # Enable autocast (mixed precision) for speed if desired
    torch.autocast(device_type='cuda', dtype=torch.float16).__enter__()

    # If using CUDA on a relatively modern GPU, enable TF-32
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------
    # Step B: Loop over all subfolders in `recordings_dir`
    # ------------------------------------------------

    # for subfolder in os.listdir(recordings_dir):
    # subfolder_path = os.path.join(recordings_dir, subfolder)
    subfolder_path = os.path.join(recordings_dir)
    if not os.path.isdir(subfolder_path):
        # continue
        exit(f"Error: {subfolder_path} is not a directory")

    # Check for "rgb/0.png"
    rgb_folder = os.path.join(subfolder_path, "rgb")
    first_frame_path = os.path.join(rgb_folder, "000000.png")
    if not os.path.isfile(first_frame_path):
        # continue  # Skip if there's no "0.png" in this folder
        exit(f"Error: {first_frame_path} does not exist")

    # ------------------------------------------------
    # Step C: Read that first frame
    # ------------------------------------------------
    img_bgr = cv2.imread(first_frame_path)
    if img_bgr is None:
        print(f"Warning: couldn't read {first_frame_path}")
        # continue
        exit(f"Error: {first_frame_path} is not a valid image")

    # Convert BGR -> RGB -> PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    # ------------------------------------------------
    # Step D: Run Grounding DINO for object detection
    # ------------------------------------------------
    inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_img.size[::-1]]  # (height, width)
    )
    # If no objects were detected, we skip or create an all-black mask
    if len(results[0]["labels"]) == 0:
        print(f"NOTE: No objects detected in {first_frame_path}")
        # Optionally create an all-black mask
        H, W = img_bgr.shape[:2]
        black_mask = np.zeros((H, W), dtype=np.uint8)
        mask_folder = os.path.join(subfolder_path, "masks")
        os.makedirs(mask_folder, exist_ok=True)
        cv2.imwrite(os.path.join(mask_folder, "0.png"), black_mask)
        # continue
        exit(f"Error: No objects detected in {first_frame_path}")

    # Extract bounding boxes from results
    input_boxes = results[0]["boxes"].cpu().numpy()  # shape: (N, 4)

    # ------------------------------------------------
    # Step E: Run SAM2 to get masks from bounding boxes
    # ------------------------------------------------
    # Tell the predictor which image we are working on
    image_predictor.set_image(np.array(pil_img.convert("RGB")))

    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False  # Single mask per bounding box
    )
    # Adjust shape if needed
    # Grounding DINO might detect multiple bounding boxes => multiple masks
    if masks.ndim == 4:
        print(f"NOTE: Multiple objects detected in {first_frame_path}")
        masks = np.squeeze(masks, axis=1)
    # Binarize each mask
    masks = masks > 0.5

    # ------------------------------------------------
    # Step F: Combine all object masks into a single B/W mask
    # ------------------------------------------------
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        combined_mask[m] = 255  # White
    # ------------------------------------------------
    # Step G: Save the black/white mask in `masks/0.png`
    # ------------------------------------------------
    mask_folder = os.path.join(subfolder_path, "masks")
    os.makedirs(mask_folder, exist_ok=True)
    mask_path = os.path.join(mask_folder, "000000.png")
    
    cv2.imwrite(mask_path, combined_mask)
    print(f"Saved mask for {first_frame_path} at {mask_path}")


if __name__ == "__main__":
    args = parse_arguments()
    process_first_frame_in_folders(
        recordings_dir=args.root_dir,
        text_prompt=args.text_prompt,
        device="cuda:0"  
    )

