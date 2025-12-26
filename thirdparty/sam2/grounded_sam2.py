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
    key_frame_idx=0,  # Add key_frame_idx parameter, default to 0
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
    sam2_predictor=None,
    detection_model="grounding-dino",  # "grounding-dino" or "florence-2"
    florence2_model=None,  # Pre-loaded Florence-2 model
    florence2_processor=None,  # Pre-loaded Florence-2 processor
    florence2_model_id="microsoft/Florence-2-large"  # Florence-2 model ID
):
    """
    Processes a folder of PNG frames to detect and segment objects based on a text prompt.

    Args:
        frames_dir (str): Path to directory containing PNG frames (e.g., 00000/, 00001/, etc.)
        text_prompt (str): Text prompt for object detection (e.g., "apple pear mango.").
        key_frame_idx (int, optional): Frame index to use for detection. 0=first frame, -1=middle frame. Defaults to 0.
        save_path (str, optional): Path to save the annotated video (e.g. ./save/).
        device (str, optional): Device to run the models on ("cuda" or "cpu"). 
        sam2_checkpoint (str, optional): Path to the SAM2 model checkpoint. 
        model_cfg (str, optional): Path to the SAM2 model configuration file. 
        grounding_model_id (str, optional): HuggingFace model ID for Grounding DINO. 
        box_threshold (float, optional): Confidence threshold for bounding boxes. 
        text_threshold (float, optional): Confidence threshold for text labels.
        fps (float, optional): FPS for output visualization video. Defaults to 12.
        detection_model (str, optional): Detection model to use ("grounding-dino" or "florence-2"). Defaults to "grounding-dino".
        florence2_model (optional): Pre-loaded Florence-2 model (used when detection_model is "florence-2").
        florence2_processor (optional): Pre-loaded Florence-2 processor (used when detection_model is "florence-2").
        florence2_model_id (str, optional): HuggingFace model ID for Florence-2. Defaults to "microsoft/Florence-2-large".

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
    
    # Get all image files (PNG or JPG), sorted by name
    png_files = sorted(
        list(frames_path.glob("*.png")) + 
        list(frames_path.glob("*.jpg")) + 
        list(frames_path.glob("*.jpeg")) +
        list(frames_path.glob("*.PNG")) +
        list(frames_path.glob("*.JPG")) +
        list(frames_path.glob("*.JPEG"))
    )
    
    if len(png_files) == 0:
        print(f"No image files (PNG/JPG) found in {frames_dir}")
        return None, None
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(str(png_files[0]))
    if first_frame is None:
        print("Failed to load first frame.")
        return None, None
    
    T = len(png_files)  # Number of frames
    H, W = first_frame.shape[:2]  # Frame height, width
    
    # Determine which frame to use for detection
    # -1 means middle frame, otherwise use the specified index
    if key_frame_idx == -1:
        actual_key_frame_idx = T // 2
    else:
        actual_key_frame_idx = key_frame_idx
    
    # Ensure the key frame index is valid
    if actual_key_frame_idx >= T or actual_key_frame_idx < 0:
        print(f"Error: Key frame index {actual_key_frame_idx} is out of range (0-{T-1})")
        return None, None
    
    # Load the key frame for detection
    key_frame = cv2.imread(str(png_files[actual_key_frame_idx]))
    if key_frame is None:
        print(f"Failed to load key frame at index {actual_key_frame_idx}.")
        return None, None
    
    # Convert key frame to RGB for detection
    key_frame_rgb = cv2.cvtColor(key_frame, cv2.COLOR_BGR2RGB)

    # ------------------------------
    # Step 2: Run object detection on a key frame
    # ------------------------------
    key_frame_pil = Image.fromarray(key_frame_rgb)
    
    if detection_model == "grounding-dino":
        # Load Grounding DINO models only if not provided (for backwards compatibility)
        if processor is None or grounding_model is None:
            print("Loading Grounding DINO model...")
            processor = AutoProcessor.from_pretrained(grounding_model_id)
            grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
        
        print(f"Running Grounding DINO detection on key frame {actual_key_frame_idx} (of {T} total frames) with prompt: '{text_prompt}'...")

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
    
    elif detection_model == "florence-2":
        # Load Florence-2 models only if not provided
        if florence2_processor is None or florence2_model is None:
            print(f"Loading Florence-2 model ({florence2_model_id})...")
            from transformers import AutoModelForCausalLM
            florence2_processor = AutoProcessor.from_pretrained(florence2_model_id, trust_remote_code=True)
            florence2_model = AutoModelForCausalLM.from_pretrained(
                florence2_model_id, 
                trust_remote_code=True, 
                attn_implementation="eager"  # Use eager attention to avoid compatibility issues
            ).eval().to(device)
        
        print(f"Running Florence-2 detection on key frame {actual_key_frame_idx} (of {T} total frames) with prompt: '{text_prompt}'...")
        
        # Use OPEN_VOCABULARY_DETECTION task for Florence-2
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        prompt = task_prompt + text_prompt
        
        # Prepare input for Florence-2
        inputs = florence2_processor(
            text=prompt,
            images=key_frame_pil,
            return_tensors="pt"
        ).to(device)
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = florence2_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
                use_cache=False  # Disable KV cache to avoid beam search issues
            )
        
        # Decode and parse results
        generated_text = florence2_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = florence2_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(key_frame_pil.width, key_frame_pil.height)
        )
        
        # Extract results from Florence-2 format
        results = parsed_answer[task_prompt]
        
        if len(results.get("bboxes", [])) == 0:
            print("No objects found by Florence-2.")
            return None, None
        
        print(f"Detection complete. Found {len(results['bboxes'])} objects.")
        
        # Parse Florence-2 detection results
        input_boxes = np.array(results["bboxes"])              # shape (N, 4)
        text_labels = results["bboxes_labels"]                  # list of N strings
        num_objects = len(input_boxes)
    
    else:
        raise ValueError(f"Unknown detection_model: {detection_model}. Must be 'grounding-dino' or 'florence-2'.")

    # Build an object ID -> name mapping from detection model's text labels
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
            frame_idx=actual_key_frame_idx,  # Use the actual key frame index
            obj_id=obj_id,
            box=box_prompt,
        )

    # Now propagate to all frames, building a 4D array: (T, O, H, W)
    masks_4d = np.zeros((T, num_objects, H, W), dtype=bool)

    # Bidirectional propagation: forward from key frame to end, backward from key frame to start
    print(f"Propagating masks bidirectionally from frame {actual_key_frame_idx}...")
    
    # Forward propagation: from key frame to end
    print(f"  Forward: frames {actual_key_frame_idx} → {T-1}")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=actual_key_frame_idx
    ):
        for j, out_obj_id in enumerate(out_obj_ids):
            if 1 <= out_obj_id <= num_objects:
                mask = (out_mask_logits[j] > 0.0).cpu().numpy()  # shape (H, W)
                masks_4d[out_frame_idx, out_obj_id - 1] = mask
    
    # Backward propagation: from key frame to beginning (only if key frame is not the first frame)
    if actual_key_frame_idx > 0:
        print(f"  Backward: frames {actual_key_frame_idx} → 0")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=actual_key_frame_idx, reverse=True
        ):
            for j, out_obj_id in enumerate(out_obj_ids):
                if 1 <= out_obj_id <= num_objects:
                    mask = (out_mask_logits[j] > 0.0).cpu().numpy()  # shape (H, W)
                    masks_4d[out_frame_idx, out_obj_id - 1] = mask
    
    print("✓ Bidirectional propagation complete")

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


def sam3_video(
    frames_dir,
    text_prompt,
    key_frame_idx=0,
    save_path=None,
    video_name=None,
    device="cuda",
    fps=12,
    sam3_predictor=None,
):
    """
    Processes a folder of PNG frames using SAM 3's unified detection + segmentation model.
    
    SAM 3 has built-in text-based detection, so no separate detection model is needed.
    It uses a session-based API for video segmentation with automatic object tracking.

    Args:
        frames_dir (str): Path to directory containing image frames (e.g., 00000/, 00001/, etc.)
        text_prompt (str): Text prompt for object detection (e.g., "person", "apple").
                          No trailing period needed (unlike Grounding DINO).
        key_frame_idx (int, optional): Frame index for initial detection. 
                                       0=first frame, -1=middle frame. Defaults to 0.
        save_path (str, optional): Path to save the annotated video (e.g. ./save/).
        video_name (str, optional): Name for the output video file.
        device (str, optional): Device to run the model on ("cuda" or "cpu").
        fps (float, optional): FPS for output visualization video. Defaults to 12.
        sam3_predictor: Pre-loaded SAM3 video predictor (required).

    Returns:
        numpy.ndarray: 4D array of masks with shape (T, O, H, W) where O is the number of unique objects.
        dict: Dictionary mapping object IDs to the text prompt used.
    """
    from pathlib import Path
    
    frames_path = Path(frames_dir)
    
    if not frames_path.exists():
        print(f"Error: Directory not found: {frames_dir}")
        return None, None
    
    if sam3_predictor is None:
        raise ValueError("sam3_predictor must be provided. Use build_sam3_video_predictor() to create one.")
    
    # Strip trailing period from text prompt if present (SAM3 doesn't need it)
    text_prompt_clean = text_prompt.rstrip('.')
    
    # Get frame files to determine dimensions and count
    png_files = sorted(
        list(frames_path.glob("*.png")) + 
        list(frames_path.glob("*.jpg")) + 
        list(frames_path.glob("*.jpeg")) +
        list(frames_path.glob("*.PNG")) +
        list(frames_path.glob("*.JPG")) +
        list(frames_path.glob("*.JPEG"))
    )
    
    if len(png_files) == 0:
        print(f"No image files (PNG/JPG) found in {frames_dir}")
        return None, None
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(str(png_files[0]))
    if first_frame is None:
        print("Failed to load first frame.")
        return None, None
    
    T = len(png_files)  # Number of frames
    H, W = first_frame.shape[:2]  # Frame height, width
    
    # Determine which frame to use for detection
    if key_frame_idx == -1:
        actual_key_frame_idx = T // 2
    else:
        actual_key_frame_idx = key_frame_idx
    
    # Ensure the key frame index is valid
    if actual_key_frame_idx >= T or actual_key_frame_idx < 0:
        print(f"Error: Key frame index {actual_key_frame_idx} is out of range (0-{T-1})")
        return None, None
    
    # ------------------------------
    # Step 1: Start SAM3 session
    # ------------------------------
    print(f"Starting SAM3 session for {frames_dir}...")
    response = sam3_predictor.handle_request({
        "type": "start_session",
        "resource_path": str(frames_dir)
    })
    session_id = response["session_id"]
    print(f"  Session ID: {session_id}")
    
    try:
        # ------------------------------
        # Step 2: Add text prompt
        # ------------------------------
        print(f"Running SAM3 detection on frame {actual_key_frame_idx} with prompt: '{text_prompt_clean}'...")
        response = sam3_predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": actual_key_frame_idx,
            "text": text_prompt_clean
        })
        
        # Check initial detection results
        initial_outputs = response.get("outputs", {})
        initial_obj_ids = initial_outputs.get("out_obj_ids", np.array([]))
        
        if len(initial_obj_ids) == 0:
            print(f"No objects detected on frame {actual_key_frame_idx}.")
            # Try middle frame if we weren't already using it
            if actual_key_frame_idx != T // 2:
                print(f"  Retrying with middle frame ({T // 2})...")
                # Reset and try again
                sam3_predictor.handle_request({
                    "type": "reset_session",
                    "session_id": session_id
                })
                response = sam3_predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": T // 2,
                    "text": text_prompt_clean
                })
                initial_outputs = response.get("outputs", {})
                initial_obj_ids = initial_outputs.get("out_obj_ids", np.array([]))
                
                if len(initial_obj_ids) == 0:
                    print("No objects detected after retry.")
                    return None, None
            else:
                return None, None
        
        print(f"  Initial detection found {len(initial_obj_ids)} objects: {initial_obj_ids.tolist()}")
        
        # ------------------------------
        # Step 3: Propagate through video
        # ------------------------------
        print(f"Propagating masks through video (direction=both)...")
        outputs_per_frame = {}
        
        for response in sam3_predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both"
        }):
            frame_idx = response["frame_index"]
            outputs_per_frame[frame_idx] = response["outputs"]
        
        print(f"  Collected outputs for {len(outputs_per_frame)} frames")
        
        # ------------------------------
        # Step 4: Convert outputs to (T, O, H, W) format
        # ------------------------------
        # Collect all unique object IDs across all frames
        all_obj_ids = set()
        for frame_idx, outputs in outputs_per_frame.items():
            obj_ids = outputs.get("out_obj_ids", np.array([]))
            all_obj_ids.update(obj_ids.tolist())
        
        if len(all_obj_ids) == 0:
            print("No objects detected in any frame.")
            return None, None
        
        # Create a sorted list of unique object IDs and a mapping
        sorted_obj_ids = sorted(all_obj_ids)
        num_objects = len(sorted_obj_ids)
        obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(sorted_obj_ids)}
        
        print(f"  Total unique objects: {num_objects}, IDs: {sorted_obj_ids}")
        
        # Create the output mask array
        masks_4d = np.zeros((T, num_objects, H, W), dtype=bool)
        
        # Fill in masks for each frame
        for frame_idx in range(T):
            if frame_idx in outputs_per_frame:
                outputs = outputs_per_frame[frame_idx]
                obj_ids = outputs.get("out_obj_ids", np.array([]))
                binary_masks = outputs.get("out_binary_masks", np.array([]))
                
                for i, obj_id in enumerate(obj_ids):
                    if obj_id in obj_id_to_idx and i < len(binary_masks):
                        idx = obj_id_to_idx[obj_id]
                        mask = binary_masks[i]
                        # Resize mask if dimensions don't match
                        if mask.shape != (H, W):
                            mask = cv2.resize(
                                mask.astype(np.uint8), 
                                (W, H), 
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        masks_4d[frame_idx, idx] = mask
        
        # Build object dictionary (map object IDs to text prompt)
        object_dict = {}
        for obj_id in sorted_obj_ids:
            idx = obj_id_to_idx[obj_id] + 1  # 1-based for consistency with gsam_video
            object_dict[idx] = text_prompt_clean
        
        print(f"✓ SAM3 propagation complete")
        print(f"  Masks shape: {masks_4d.shape}")
        print(f"  Objects detected: {num_objects}")
        
        # ------------------------------
        # Step 5 (Optional): Save annotated video
        # ------------------------------
        if save_path is not None:
            output_filename = f'{video_name}.mp4' if video_name else 'sam3.mp4'
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(os.path.join(save_path, output_filename), fourcc, fps, (W, H))
            
            for fidx in range(T):
                frame_bgr = cv2.imread(str(png_files[fidx]))
                frame_bgr_annot = frame_bgr.copy()
                
                # Collect all valid masks
                valid_masks = []
                valid_obj_ids = []
                for obj_idx in range(num_objects):
                    mask_bool = masks_4d[fidx, obj_idx]
                    if mask_bool.any():
                        valid_masks.append(mask_bool)
                        valid_obj_ids.append(obj_idx + 1)  # 1-based
                
                if len(valid_masks) == 0:
                    out_video.write(frame_bgr_annot)
                    continue
                
                final_masks = np.stack(valid_masks, axis=0)
                xyxy = sv.mask_to_xyxy(final_masks)
                
                detections = sv.Detections(
                    xyxy=xyxy,
                    mask=final_masks,
                    class_id=np.array(valid_obj_ids, dtype=np.int32)
                )
                
                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=frame_bgr_annot, detections=detections)
                
                label_annotator = sv.LabelAnnotator()
                label_texts = [object_dict[obj_id] for obj_id in valid_obj_ids]
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=label_texts
                )
                
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                
                out_video.write(annotated_frame)
            
            out_video.release()
            print(f"Annotated video saved at: {os.path.join(save_path, output_filename)}")
        
        return masks_4d, object_dict
        
    finally:
        # ------------------------------
        # Step 6: Close session to free GPU memory
        # ------------------------------
        sam3_predictor.handle_request({
            "type": "close_session",
            "session_id": session_id
        })
        print(f"  Session {session_id} closed")


def gsam(
    video_path, 
    text, 
    save_dir=None,
    device="cuda",
    sam2_checkpoint="sam2/checkpoints/sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    grounding_model_id="IDEA-Research/grounding-dino-base",
    box_threshold=0.5,
    text_threshold=0.5
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