import os
import sys
import subprocess
import urllib.request
import zipfile

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def run_command(cmd, cwd=None, env=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, env=env, check=True)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. GroundingDINO
    gdino_dir = os.path.join(models_dir, "groundingdino")
    os.makedirs(gdino_dir, exist_ok=True)

    gdino_config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_config_dest = os.path.join(gdino_dir, "GroundingDINO_SwinT_OGC.py")
    if not os.path.exists(gdino_config_dest):
        download_file(gdino_config_url, gdino_config_dest)

    gdino_weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    gdino_weights_dest = os.path.join(gdino_dir, "groundingdino_swint_ogc.pth")
    if not os.path.exists(gdino_weights_dest):
        download_file(gdino_weights_url, gdino_weights_dest)

    try:
        import groundingdino
        print("GroundingDINO is already installed.")
    except ImportError:
        print("Installing GroundingDINO from github...")
        # Disable CUDA extension build (requires MSVC) and force use of existing torch
        env = os.environ.copy()
        env["GROUNDINGDINO_BUILD_CUDA_EXT"] = "0"
        run_command("pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git", env=env)

    # 2. SAM2
    sam2_dir = os.path.join(models_dir, "sam2")
    os.makedirs(sam2_dir, exist_ok=True)

    sam2_weights_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt"
    sam2_weights_dest = os.path.join(sam2_dir, "sam2.1_hiera_tiny.pt")
    if not os.path.exists(sam2_weights_dest):
        download_file(sam2_weights_url, sam2_weights_dest)

    sam2_repo_dir = os.path.join(sam2_dir, "repo")
    if not os.path.exists(sam2_repo_dir):
        print("Installing SAM2 from github...")
        run_command(f"git clone https://github.com/facebookresearch/sam2.git {sam2_repo_dir}")
        env = os.environ.copy()
        env["SAM2_BUILD_CUDA"] = "0"
        run_command("pip install --no-build-isolation -e .", cwd=sam2_repo_dir, env=env)

    # 3. RealESRGAN
    resrgan_weights_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    resrgan_weights_dest = os.path.join(models_dir, "RealESRGAN_x4plus.pth")
    if not os.path.exists(resrgan_weights_dest):
        download_file(resrgan_weights_url, resrgan_weights_dest)

    try:
        import realesrgan
    except ImportError:
        print("Installing RealESRGAN...")
        run_command("pip install basicsr facexlib")
        run_command("pip install git+https://github.com/xinntao/Real-ESRGAN.git")

    print("\nModel setup complete! All required weights and code are localized in 'models/'")

if __name__ == "__main__":
    main()
