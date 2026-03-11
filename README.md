# RotoBot — Ultimate Automatic Rotoscoping Tool

RotoBot is a program that uses 3 types of AI to automate rotoscoping of images, videos, and batches into files with alpha channel

Automatic foreground extraction and alpha channel generation for still images and videos,
powered by **Grounded SAM2** (GroundingDINO + SAM2).

# FILE: README_LICENSE_SECTION.md | ID: QWERTYUIOP
## Licensing and Commercial Use

**Rotobot is dual-licensed.**

This software is fundamentally free for open-source, academic, and individual use under the terms of the **GNU Affero General Public License v3.0 (AGPLv3)**. You are encouraged to use, modify, and distribute this tool to save time and build your own personal projects. 

However, the AGPLv3 contains strong copyleft requirements. If you intend to use Rotobot within a proprietary, closed-source corporate environment, or offer its functionality as a service over a network without open-sourcing your entire backend infrastructure, you are not permitted to use the AGPLv3 license.

**Corporate / Commercial Exemption:**
To legally bypass the AGPLv3 requirements and integrate Rotobot into your commercial operations, you must purchase a Commercial License. 

* **Individuals & Open Source:** Free (AGPLv3)
* **Commercial & Enterprise:** [Link to your email / payment portal / contact form]

By maintaining this dual-license structure, I can continue to fund the development of this tool for the community while ensuring corporations pay their fair share for the engineering leverage it provides.

Rotobot is completely portable and can manage its own models and dependencies, keeping your system clean.

## What It Does

Takes normal images (JPG, PNG, BMP, etc.) and videos (MP4, WebM, etc.) and produces production-ready **RGBA PNG files or WebM files**
with clean alpha channels — no green screen required. Uses AI-powered segmentation
that is vastly superior to conventional keying.

---

## 🚀 Installation Instructions
Before installing, please ensure you have an **NVIDIA GPU (Minimum 6GB VRAM)**, **Python (3.10 or 3.11)**, and **Git**. 
*Note: During python installation, you MUST check the box that says "Add python.exe to PATH".*

1. **Extract the Folder**: Unzip the Rotobot folder and place it wherever you want to keep the program on your PC.
2. **Run the Installer**: Inside the folder, double-click the file named `install_rotobot.bat`.
3. **Wait**: A command prompt window will open. It will automatically:
   - Create a private, isolated Python environment just for this program.
   - Download the massive PyTorch libraries.
   - Clone and install the SAM2 and GroundingDINO repositories.
   - Download all necessary multi-gigabyte AI model weights into a local `models/` folder.
   *(This step can take 10-20 minutes depending on internet speed. Let it run until you see "INSTALLATION COMPLETE")*
4. **Close**: Press any key to close the window when it successfully finishes.

---

## 🖥️ Running the Application

1. Double-click `run_rotobot.bat`.
2. The User Interface will open automatically. Enjoy!

### 💡 Running on Lower-End GPUs (6GB - 12GB VRAM)
If your graphics card struggles with "Out of Memory" crashes:
1. Look at the bottom settings box in the interface.
2. Check the box that says **"Optimize for VRAM (Run in Serial)"**.
3. Set the **VRAM Limit (GB)** box to match your graphics card's maximum VRAM (e.g., set to 8 if you have an RTX 3060 Ti).
4. The program will now actively swap the models in and out of system RAM while it runs, keeping VRAM usage safe at the cost of slightly slower processing times!

---

## Quick Start (CLI)

### CLI — Single Image
```bash
python rotobot.py --input photo.jpg --prompt "person"
```

### CLI — Batch Processing
```bash
python rotobot.py --input ./photos/ --prompt "person . dog" --output ./alphas/
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | *required* | Input image file or folder |
| `--output`, `-o` | same as input | Output file or folder |
| `--prompt`, `-p` | `"foreground subject . main object"` | What to extract |
| `--threshold`, `-t` | `0.25` | Detection confidence (0–1) |
| `--refine`, `-r` | `1` | Edge smoothing (0=off, 1=light, 3=heavy) |
| `--invert` | off | Invert alpha (keep background) |
| `--vram-optimize` | off | Run models in serial to save VRAM |
| `--vram-limit` | `11` | Set VRAM limit in GB |

## Prompt Tips

- Separate multiple object types with `.`: `"person . dog . cat"`
- Be specific: `"standing person"` vs `"person"`
- Common prompts: `"person"`, `"animal"`, `"car"`, `"building"`, `"foreground subject"`

## Architecture Updates
```
Rotobot/
├── install_rotobot.bat # 1-click Windows installer
├── setup_models.py     # Script to download weights & repos
├── run_rotobot.bat     # Windows launcher
├── rotobot.py          # CLI entry point
├── rotobot_gui.py      # PyQt6 drag-and-drop GUI
├── rotobot_engine.py   # Core engine (GroundingDINO + SAM2)
├── rotobot_vision.py   # Vision extraction (Uses LLaMA Vision)
├── requirements.txt    # Dependencies
└── README.md           # This file
```
Models are stored locally inside `Rotobot/models/` when you run the installation script.
