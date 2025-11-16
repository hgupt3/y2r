#!/usr/bin/env python3
"""
Script to download pretrained models for DiffuEraser.
This will download models from Hugging Face.
"""

import os
import subprocess
import sys

def run_command(cmd):
    """Run a shell command and print output."""
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: Command failed with return code {result.returncode}")
    return result.returncode

def main():
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    os.chdir(weights_dir)
    
    print("=" * 70)
    print("DiffuEraser Model Download Script")
    print("=" * 70)
    
    # Check if git-lfs is installed
    print("\n[1/5] Checking git-lfs installation...")
    ret = subprocess.run("git lfs version", shell=True, capture_output=True)
    if ret.returncode != 0:
        print("git-lfs not found. Installing...")
        run_command("sudo apt-get update && sudo apt-get install -y git-lfs")
        run_command("git lfs install")
    else:
        print("git-lfs is already installed.")
    
    # Download DiffuEraser weights
    print("\n[2/5] Downloading DiffuEraser weights...")
    if not os.path.exists("diffuEraser"):
        run_command("git clone https://huggingface.co/lixiaowen/diffuEraser")
    else:
        print("diffuEraser directory already exists, skipping.")
    
    # Download stable-diffusion-v1-5 (minimal files only to save space)
    print("\n[3/5] Downloading stable-diffusion-v1-5 (minimal files)...")
    if not os.path.exists("stable-diffusion-v1-5"):
        os.makedirs("stable-diffusion-v1-5", exist_ok=True)
        os.chdir("stable-diffusion-v1-5")
        
        # Initialize sparse checkout to download only needed files
        run_command("git init")
        run_command("git remote add origin https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5")
        run_command("git config core.sparseCheckout true")
        
        # Specify which folders/files to download
        with open(".git/info/sparse-checkout", "w") as f:
            f.write("feature_extractor/\n")
            f.write("model_index.json\n")
            f.write("safety_checker/\n")
            f.write("scheduler/\n")
            f.write("text_encoder/\n")
            f.write("tokenizer/\n")
        
        run_command("git pull origin main")
        os.chdir("..")
    else:
        print("stable-diffusion-v1-5 directory already exists, skipping.")
    
    # Download PCM_Weights
    print("\n[4/5] Downloading PCM_Weights...")
    if not os.path.exists("PCM_Weights"):
        run_command("git clone https://huggingface.co/wangfuyun/PCM_Weights")
    else:
        print("PCM_Weights directory already exists, skipping.")
    
    # Download sd-vae-ft-mse
    print("\n[5/5] Downloading sd-vae-ft-mse...")
    if not os.path.exists("sd-vae-ft-mse"):
        run_command("git clone https://huggingface.co/stabilityai/sd-vae-ft-mse")
    else:
        print("sd-vae-ft-mse directory already exists, skipping.")
    
    # Download ProPainter weights
    print("\n[Bonus] Downloading ProPainter weights...")
    if not os.path.exists("propainter"):
        os.makedirs("propainter", exist_ok=True)
        os.chdir("propainter")
        
        # Download the three required files
        files = [
            ("ProPainter.pth", "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth"),
            ("raft-things.pth", "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth"),
            ("recurrent_flow_completion.pth", "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth")
        ]
        
        for filename, url in files:
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                run_command(f"wget {url}")
            else:
                print(f"{filename} already exists, skipping.")
        
        os.chdir("..")
    else:
        print("propainter directory already exists, skipping.")
    
    print("\n" + "=" * 70)
    print("Download complete!")
    print("=" * 70)
    print("\nYou can now run the demo with:")
    print("  python run_diffueraser.py")
    print("\nOr try different examples:")
    print("  python run_diffueraser.py --input_video examples/example1/video.mp4 --input_mask examples/example1/mask.mp4")

if __name__ == "__main__":
    main()

