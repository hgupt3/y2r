from sam2.grounded_sam2 import gsam, gsam_video
from cotracker3.tap import cotracker
import numpy as np

video_path = "/home/harsh/sam/test/apple.mp4"
text = "apple."
out_path = "/home/harsh/sam/test/out"
device = 'cuda:1'

print('finding objects in video...')
masks_4d, obj_dict = gsam_video(
    video_path=video_path,
    text_prompt=text,
    save_path=out_path,  
    device=device
)

combined_mask = np.any(masks_4d, axis=1).astype(float)[0][None, None]

print('tracking objects in video...')
cotracker(video_path, 
          combined_mask, 
          grid_size=30, 
          save_dir=out_path,
          device=device
)
