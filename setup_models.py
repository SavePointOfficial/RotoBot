import os
import sys
import subprocess
import urllib.request
import time


def download_file(url, dest_path, fallback_url=None):
    """Download a file with retry logic and optional fallback URL."""
    for attempt in range(3):
        try:
            print(f"  Downloading {os.path.basename(dest_path)}...")
            print(f"    URL: {url}")
            # Add User-Agent header to avoid 403 from servers that block bare urllib
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Rotobot-Installer)"
            })
            with urllib.request.urlopen(req, timeout=120) as response:
                total = response.headers.get("Content-Length")
                total = int(total) if total else None
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks

                with open(dest_path + ".tmp", "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            mb = downloaded / (1024 * 1024)
                            total_mb = total / (1024 * 1024)
                            print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

            # Rename on success (atomic-ish)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(dest_path + ".tmp", dest_path)
            print(f"\n    Download complete ({os.path.getsize(dest_path) / (1024*1024):.1f} MB)")
            return True

        except Exception as e:
            # Clean up partial file
            tmp = dest_path + ".tmp"
            if os.path.exists(tmp):
                os.remove(tmp)

            if attempt < 2:
                wait = (attempt + 1) * 5
                print(f"\n    Attempt {attempt + 1} failed: {e}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n    All attempts failed for primary URL: {e}")

    # Try fallback URL if provided
    if fallback_url:
        print(f"    Trying fallback URL: {fallback_url}")
        return download_file(fallback_url, dest_path, fallback_url=None)

    return False


def run_command(cmd, cwd=None, env=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, env=env, check=True)


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    errors = []

    # =========================================================================
    # 1. GroundingDINO
    # =========================================================================
    print("\n" + "=" * 60)
    print("  [1/3] GroundingDINO")
    print("=" * 60)

    gdino_dir = os.path.join(models_dir, "groundingdino")
    os.makedirs(gdino_dir, exist_ok=True)

    gdino_config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_config_dest = os.path.join(gdino_dir, "GroundingDINO_SwinT_OGC.py")
    if not os.path.exists(gdino_config_dest):
        if not download_file(gdino_config_url, gdino_config_dest):
            errors.append("GroundingDINO config download failed")

    gdino_weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    gdino_weights_dest = os.path.join(gdino_dir, "groundingdino_swint_ogc.pth")
    if not os.path.exists(gdino_weights_dest):
        if not download_file(gdino_weights_url, gdino_weights_dest):
            errors.append("GroundingDINO weights download failed")

    try:
        import groundingdino
        print("GroundingDINO package is already installed.")
    except ImportError:
        print("Installing GroundingDINO from github (CPU-only, no CUDA extensions)...")
        # Clone the repo, then patch setup.py to remove the CUDA C++ extension
        # build entirely.  Env-var tricks don't work because PyTorch's
        # cpp_extension resolves CUDA_HOME from nvcc on PATH and raises a
        # version-mismatch error before GroundingDINO's own flag is checked.
        gdino_repo_dir = os.path.join(gdino_dir, "repo")
        if not os.path.exists(gdino_repo_dir):
            run_command(f"git clone https://github.com/IDEA-Research/GroundingDINO.git {gdino_repo_dir}")

        # Patch setup.py: strip ext_modules and cmdclass so it becomes a
        # pure-Python install with zero C++/CUDA compilation.
        gdino_setup_py = os.path.join(gdino_repo_dir, "setup.py")
        with open(gdino_setup_py, "r") as f:
            setup_src = f.read()
        setup_src = setup_src.replace(
            'ext_modules=get_extensions(),',
            '# ext_modules removed for CPU-only install'
        )
        setup_src = setup_src.replace(
            'cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},',
            '# cmdclass removed for CPU-only install'
        )
        with open(gdino_setup_py, "w") as f:
            f.write(setup_src)
        print("  -> Patched setup.py to remove CUDA extensions")

        # Patch ms_deform_attn.py: make it fall back to pure-Python when _C
        # is not available (instead of crashing at inference time).
        deform_attn_py = os.path.join(
            gdino_repo_dir, "groundingdino", "models", "GroundingDINO", "ms_deform_attn.py")
        if os.path.exists(deform_attn_py):
            with open(deform_attn_py, "r") as f:
                attn_src = f.read()
            # Add _C_available flag on import
            attn_src = attn_src.replace(
                'from groundingdino import _C\nexcept:',
                'from groundingdino import _C\n    _C_available = True\nexcept:'
            )
            attn_src = attn_src.replace(
                'warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")',
                'warnings.warn("Failed to load custom C++ ops. Using pure-Python fallback.")\n    _C_available = False'
            )
            # Make forward() check _C_available before using the extension
            attn_src = attn_src.replace(
                'if torch.cuda.is_available() and value.is_cuda:',
                'if _C_available and torch.cuda.is_available() and value.is_cuda:'
            )
            with open(deform_attn_py, "w") as f:
                f.write(attn_src)
            print("  -> Patched ms_deform_attn.py for _C fallback")

        run_command("pip install --no-build-isolation -e .", cwd=gdino_repo_dir)

    # =========================================================================
    # 2. SAM2
    # =========================================================================
    print("\n" + "=" * 60)
    print("  [2/3] SAM2")
    print("=" * 60)

    sam2_dir = os.path.join(models_dir, "sam2")
    os.makedirs(sam2_dir, exist_ok=True)

    # SAM 2.1 weights — primary URL updated from /072824/ to /092824/ per
    # official repo (https://github.com/facebookresearch/sam2).
    # Fallback to Hugging Face if the primary URL is blocked.
    sam2_weights_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    sam2_weights_fallback = "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt"
    sam2_weights_dest = os.path.join(sam2_dir, "sam2.1_hiera_tiny.pt")
    if not os.path.exists(sam2_weights_dest):
        if not download_file(sam2_weights_url, sam2_weights_dest,
                             fallback_url=sam2_weights_fallback):
            errors.append("SAM2 weights download failed")

    sam2_repo_dir = os.path.join(sam2_dir, "repo")
    if not os.path.exists(sam2_repo_dir):
        print("Installing SAM2 from github...")
        run_command(f"git clone https://github.com/facebookresearch/sam2.git {sam2_repo_dir}")
        env = os.environ.copy()
        env["SAM2_BUILD_CUDA"] = "0"
        env["FORCE_CUDA"] = "0"
        env.pop("CUDA_HOME", None)
        env.pop("CUDA_PATH", None)
        run_command("pip install --no-build-isolation -e .", cwd=sam2_repo_dir, env=env)

    # =========================================================================
    # 3. RealESRGAN
    # =========================================================================
    print("\n" + "=" * 60)
    print("  [3/3] RealESRGAN")
    print("=" * 60)

    resrgan_weights_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    resrgan_weights_dest = os.path.join(models_dir, "RealESRGAN_x4plus.pth")
    if not os.path.exists(resrgan_weights_dest):
        if not download_file(resrgan_weights_url, resrgan_weights_dest):
            errors.append("RealESRGAN weights download failed")

    try:
        import realesrgan
        print("RealESRGAN is already installed.")
    except ImportError:
        print("Installing RealESRGAN...")
        run_command("pip install basicsr facexlib")
        run_command("pip install git+https://github.com/xinntao/Real-ESRGAN.git")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    if errors:
        print("=" * 60)
        print("  INSTALLATION FAILED -- the following downloads failed:")
        for err in errors:
            print(f"    [FAIL] {err}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("=" * 60)
        print("  MODEL SETUP COMPLETE [OK]")
        print("  All weights and code are localized in 'models/'")
        print("=" * 60)


if __name__ == "__main__":
    main()
