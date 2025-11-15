import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def gsam_video(
    frames_dir,
    text_prompt,
    save_path=None,
    video_name=None,
    device="cuda",
    sam2_checkpoint="sam2/checkpoints/sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    grounding_model_id="IDEA-Research/grounding-dino-base",
    box_threshold=0.3,
    text_threshold=0.3,
    fps=12,
    grounding_model=None,  
    processor=None,  
    sam2_predictor=None 
):
    """
    Processes a folder of PNG frames to detect and segment objects based on a text prompt.

    Args:
        frames_dir (str): Path to directory containing PNG frames (e.g., 00000/, 00001/, etc.)
        text_prompt (str): Text prompt for object detection (e.g., "apple pear mango.").
        save_path (str, optional): Path to save the annotated video (e.g. ./save/).
        device (str, optional): Device to run the models on ("cuda" or "cpu"). 
        sam2_checkpoint (str, optional): Path to the SAM2 model checkpoint. 
        model_cfg (str, optional): Path to the SAM2 model configuration file. 
        grounding_model_id (str, optional): HuggingFace model ID for Grounding DINO. 
        box_threshold (float, optional): Confidence threshold for bounding boxes. 
        text_threshold (float, optional): Confidence threshold for text labels.
        fps (float, optional): FPS for output visualization video. Defaults to 12.

    Returns:
        numpy.ndarray: 4D array of masks with shape (T, O, H, W).
        dict: Dictionary mapping object IDs to object names.
    """

    # ------------------------------
    # Step 1: Read PNG frames from directory
    # ------------------------------
    from pathlib import Path
    frames_path = Path(frames_dir)
    
    if not frames_path.exists():
        print(f"Error: Directory not found: {frames_dir}")
        return None, None
    
    # Get all PNG files, sorted by name
    png_files = sorted(frames_path.glob("*.png"))
    
    if len(png_files) == 0:
        print(f"No PNG files found in {frames_dir}")
        return None, None
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(str(png_files[0]))
    if first_frame is None:
        print("Failed to load first frame.")
        return None, None
    
    T = len(png_files)  # Number of frames
    H, W = first_frame.shape[:2]  # Frame height, width
    
    # Convert first frame to RGB for detection
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # ------------------------------
    # Step 2: Run Grounding DINO on a key frame
    # ------------------------------
    # Load models only if not provided (for backwards compatibility)
    if processor is None or grounding_model is None:
        print("Loading Grounding DINO model...")
        processor = AutoProcessor.from_pretrained(grounding_model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    
    print(f"Running object detection on key frame with prompt: '{text_prompt}'...")
    key_frame_idx = 0
    key_frame_pil = Image.fromarray(first_frame_rgb)

    # Prepare input for Grounding DINO
    inputs = processor(
        images=key_frame_pil,
        text=text_prompt,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    
    # Post-process the results to get bounding boxes and text labels
    # Note: post_process_grounded_object_detection API changed in transformers 4.57+
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[key_frame_pil.size[::-1]],  # (height, width)
        threshold=box_threshold  # Combined threshold parameter in newer versions
    )
    
    print(f"Detection complete. Found {len(results[0]['boxes'])} objects.")
    
    if len(results[0]["boxes"]) == 0:
        print("No objects found by Grounding DINO.")
        return None, None
    
    # Boxes (Nx4) and text labels (list of strings)
    input_boxes = results[0]["boxes"].cpu().numpy()        # shape (N, 4)
    text_labels = results[0]["labels"]                     # list of N strings
    num_objects = len(input_boxes)

    # Build an object ID -> name mapping from DINO's text labels
    # e.g. {1: "apple", 2: "apple", 3: "pear", ...}
    object_dict = {}
    for i, label_str in enumerate(text_labels):
        obj_id = i + 1  # 1-based
        object_dict[obj_id] = label_str

    # ------------------------------
    # Step 3: Build the SAM 2 video predictor and init state
    # ------------------------------
    # Load SAM2 predictor only if not provided
    if sam2_predictor is None:
        print("Building SAM2 video predictor...")
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    else:
        predictor = sam2_predictor
    
    # Enable autocast for better performance with Flash Attention
    # Must be done BEFORE init_state to affect model inference
    autocast_context = None
    if device.startswith("cuda"):
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        autocast_context.__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    print("Initializing SAM2 inference state...")
    # Initialize inference state with directory path (SAM2 will load frames from directory)
    inference_state = predictor.init_state(video_path=frames_dir)
    print("SAM2 initialized successfully.")

    # ------------------------------
    # Step 4: Add bounding box prompts & propagate
    # ------------------------------
    # Add each bounding box to the key frame
    for i, box in enumerate(input_boxes):
        obj_id = i + 1
        box_prompt = box.astype(np.float32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=key_frame_idx,
            obj_id=obj_id,
            box=box_prompt,
        )

    # Now propagate to all frames, building a 4D array: (T, O, H, W)
    masks_4d = np.zeros((T, num_objects, H, W), dtype=bool)

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        for j, out_obj_id in enumerate(out_obj_ids):
            if 1 <= out_obj_id <= num_objects:
                mask = (out_mask_logits[j] > 0.0).cpu().numpy()  # shape (H, W)
                masks_4d[out_frame_idx, out_obj_id - 1] = mask

    # ------------------------------
    # Step 5 (Optional): Save annotated video
    # ------------------------------
    if save_path is not None:
        # Use video_name if provided, otherwise default to 'gsam.mp4'
        output_filename = f'{video_name}.mp4' if video_name else 'gsam.mp4'
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(os.path.join(save_path, output_filename), fourcc, fps, (W, H))

        for fidx in range(T):
            # Load frame from disk
            frame_bgr = cv2.imread(str(png_files[fidx]))
            frame_bgr_annot = frame_bgr.copy()

            # Collect all valid masks (the ones with any True pixels)
            # We'll store them in a list to stack later.
            valid_masks = []
            valid_obj_ids = []
            for obj_idx in range(num_objects):
                mask_bool = masks_4d[fidx, obj_idx]  # shape (H, W), bool
                if mask_bool.any():
                    valid_masks.append(mask_bool)
                    # object IDs are 1-based in object_dict
                    valid_obj_ids.append(obj_idx + 1)

            # If no objects found in this frame, just write the original
            if len(valid_masks) == 0:
                out_video.write(frame_bgr_annot)
                continue

            # Stack all valid masks into shape (N, H, W),
            # where N = number of detected objects in this frame
            final_masks = np.stack(valid_masks, axis=0)  # shape (N, H, W)

            # Convert each mask to an (N,4) bounding box
            xyxy = sv.mask_to_xyxy(final_masks)

            # Build a single Detections object for this frame
            detections = sv.Detections(
                xyxy=xyxy,                  # (N, 4)
                mask=final_masks,           # (N, H, W)
                class_id=np.array(valid_obj_ids, dtype=np.int32)  # (N,)
            )

            # Annotate using your style
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(
                scene=frame_bgr_annot, 
                detections=detections
            )

            label_annotator = sv.LabelAnnotator()
            # Build the label list from your object_dict
            label_texts = [object_dict[obj_id] for obj_id in valid_obj_ids]
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=label_texts
            )

            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )

            # Write the annotated frame into the output video
            out_video.write(annotated_frame)

        out_video.release()
        output_filename = f'{video_name}.mp4' if video_name else 'gsam.mp4'
        print(f"Annotated video saved at: {os.path.join(save_path, output_filename)}")

    # ------------------------------
    # Step 6: Return the results
    # ------------------------------
    # Clean up to prevent memory leaks
    if autocast_context is not None:
        autocast_context.__exit__(None, None, None)
    
    # Explicitly delete inference state to free memory
    del inference_state
    
    # `masks_4d` has shape (T, O, H, W)
    # `object_dict`: e.g. {1: "apple", 2: "apple", 3: "pear", 4: "pear"}
    return masks_4d, object_dict

def gsam(
    video_path, 
    text, 
    save_dir=None,
    device="cuda",
    sam2_checkpoint="sam2/checkpoints/sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    grounding_model_id="IDEA-Research/grounding-dino-base",
    box_threshold=0.3,
    text_threshold=0.3
    ):
    """
    Processes a single frame from a video to detect and segment objects based on a text prompt.
    
    Args:
        video_path (str): Path to the input video file.
        text (str): Text prompt for object detection (e.g., "apple pear mango.").
        save_dir (str, optional): Directory path to save the annotated image. 
        device (str, optional): Device to run the models on ("cuda" or "cpu"). 
        sam2_checkpoint (str, optional): Path to the SAM2 model checkpoint. 
        model_cfg (str, optional): Path to the SAM2 model configuration file. 
        grounding_model_id (str, optional): HuggingFace model ID for Grounding DINO. 
        box_threshold (float, optional): Confidence threshold for bounding boxes. 
        text_threshold (float, optional): Confidence threshold for text labels. 
    
    Returns:
        numpy.ndarray: Array of masks for each detected object as (O, H, W).
        dict: Mapping from object IDs to object labels.
    """
    
    # -----------------------------------
    # Step 1: Read the first frame from the video
    # -----------------------------------
    video = cv2.VideoCapture(video_path)
    ret, img = video.read()
    
    if not ret:
        print(f"Failed to read the video: {video_path}")
        return None, None

    # Enable automatic mixed precision with bfloat16 for faster computation
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    
    # If using CUDA and the GPU supports TensorFloat-32 (e.g., NVIDIA Ampere architecture), enable it
    if device.startswith("cuda") and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # -----------------------------------
    # Step 2: Initialize SAM2 Image Predictor
    # -----------------------------------

    # Build SAM2 model and create an image predictor instance
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # -----------------------------------
    # Step 3: Initialize Grounding DINO Model from HuggingFace
    # -----------------------------------
    
    # Load the processor and model for zero-shot object detection
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)

    # -----------------------------------
    # Step 4: Preprocess the Image
    # -----------------------------------
    # Convert the BGR image (from OpenCV) to RGB and then to PIL format
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # -----------------------------------
    # Step 5: Run Grounding DINO on the Image
    # -----------------------------------
    # Prepare inputs for Grounding DINO with the image and text prompt
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    # Perform forward pass without computing gradients
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # Post-process the outputs to obtain bounding boxes and labels
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,   # Confidence threshold for boxes
        text_threshold=text_threshold,  # Confidence threshold for text labels
        target_sizes=[image.size[::-1]]  # (height, width)
    )

    image_predictor.set_image(np.array(image.convert("RGB")))
    
    # Extract bounding boxes and object labels from Grounding DINO results
    input_boxes = results[0]["boxes"].cpu().numpy()  # Shape: (N, 4) where N is number of detections
    OBJECTS = results[0]["labels"]                   # List of N object labels

    # If no objects are detected, return early
    if len(OBJECTS) == 0:
        print("No objects detected by Grounding DINO.")
        return None, None

    # -----------------------------------
    # Step 6: Generate Masks Using SAM2
    # -----------------------------------
    # Use SAM2 to predict masks for the detected bounding boxes
    masks, scores, logits = image_predictor.predict(
        point_coords=None,       # No point prompts
        point_labels=None,       # No point labels
        box=input_boxes,         # Bounding box prompts
        multimask_output=False,  # Single mask per box
    )

    # Adjust mask dimensions if multiple objects are detected
    if len(OBJECTS) != 1:
        # If masks have shape (1, H, W), add a new axis to make it (N, H, W)
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        # If masks have shape (N, 1, H, W), squeeze the second dimension to get (N, H, W)
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
    
    # Binarize the masks with a threshold of 0.5
    masks = masks > 0.5
    
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    
    # -----------------------------------
    # Step 7: Annotate and Save the Image (Optional)
    # -----------------------------------
    if save_dir is not None:
        # Generate a list of object IDs
        object_ids = [idx for idx in range(1, len(OBJECTS) + 1)]
        
        # Create a Detections object with bounding boxes, masks, and class IDs
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # Convert masks to bounding box coordinates (N, 4)
            mask=masks,                    # Boolean masks for each object (N, H, W)
            class_id=np.array(object_ids, dtype=np.int32),  # Class IDs as integers (N,)
        )
        
        # Initialize annotators from Supervision
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        # Create a copy of the original image to annotate
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame, 
            detections=detections, 
            labels=[ID_TO_OBJECTS[i] for i in object_ids]
        )
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # Save the annotated image to the specified directory
        cv2.imwrite(os.path.join(save_dir, 'gsam.png'), annotated_frame)
        print(f"Annotated image saved at: {os.path.join(save_dir, 'gsam.png')}")
    
    # -----------------------------------
    # Step 8: Return Results
    # -----------------------------------
    # Return the mapping of object IDs to labels and the corresponding masks
    return masks, ID_TO_OBJECTS
    
if __name__ == "__main__":
    video_path = "/home/harsh/sam/apple.mp4"
    text = "apple."
    out_video_path = "/home/harsh/sam/save/apple.mp4"

    masks_4d, obj_dict = gsam_video(
        video_path=video_path,
        text_prompt=text,
        save_path=out_video_path,   # single MP4 output
        device="cuda"
    )

    print("masks_4d shape:", None if masks_4d is None else masks_4d.shape)
    if obj_dict is not None:
        for obj_id, name in obj_dict.items():
            print(f"Object ID {obj_id} -> {name}")