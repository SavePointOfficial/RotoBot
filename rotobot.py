#!/usr/bin/env python
"""
Rotobot — Automatic Rotoscoping CLI

Usage:
    python rotobot.py --input photo.jpg --prompt "person"
    python rotobot.py --input ./photos/ --prompt "person . dog" --output ./alphas/
    python rotobot.py --input photo.jpg --prompt "car" --refine 2 --invert
    python rotobot.py --input table.jpg --auto --output ./elements/
"""

import os
import sys
import time
import argparse
import glob

# Resolve imports
ROTOBOT_DIR = os.path.dirname(os.path.abspath(__file__))
SANDBOX_ROOT = os.path.dirname(ROTOBOT_DIR)
if ROTOBOT_DIR not in sys.path:
    sys.path.insert(0, ROTOBOT_DIR)

from rotobot_logging import get_logger
log = get_logger("CLI")


# Supported extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


def get_media_files(path: str) -> list:
    """Get list of image/video files from a path (file or directory)."""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = []
        for f in sorted(os.listdir(path)):
            ext = os.path.splitext(f)[1].lower()
            if ext in MEDIA_EXTS:
                files.append(os.path.join(path, f))
        return files
    else:
        # Try as glob
        return sorted(glob.glob(path))


def make_output_path(input_path: str, output_arg: str) -> str:
    """Determine the output file path."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_name = base + "_alpha.png"

    if output_arg:
        if os.path.isdir(output_arg) or output_arg.endswith(os.sep):
            return os.path.join(output_arg, out_name)
        else:
            # If processing a single file, use output_arg as-is
            return output_arg
    else:
        # Same directory as input
        return os.path.join(os.path.dirname(input_path), out_name)


def main():
    parser = argparse.ArgumentParser(
        description="Rotobot — Automatic Rotoscoping Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rotobot.py --input photo.jpg --prompt "person"
  python rotobot.py --input ./photos/ --prompt "person . dog" --output ./alphas/
  python rotobot.py --input photo.jpg --prompt "car" --refine 2 --invert
  python rotobot.py --input table.jpg --auto --output ./elements/

Prompt tips:
  Use "." to separate multiple object types: "person . dog . cat"
  Be specific for best results: "standing person" vs just "person"
  Common prompts: "person", "animal", "car", "building", "foreground subject"
        """
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Input image file or folder of images")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file or folder (default: <input>_alpha.png)")
    parser.add_argument("--prompt", "-p", default="foreground subject . main object",
                        help="What to extract (use . to separate concepts)")
    parser.add_argument("--threshold", "-t", type=float, default=0.25,
                        help="Detection confidence threshold (0-1, default: 0.25)")
    parser.add_argument("--text-threshold", type=float, default=0.20,
                        help="Text matching threshold (0-1, default: 0.20)")
    parser.add_argument("--refine", "-r", type=int, default=1,
                        choices=[0, 1, 2, 3],
                        help="Edge refinement strength (0=off, 1=light, 3=heavy)")
    parser.add_argument("--invert", action="store_true",
                        help="Invert alpha (keep background instead of subject)")
    parser.add_argument("--auto", "--inventory", action="store_true",
                        dest="auto_inventory",
                        help="Auto-inventory mode: use Vision AI to detect all "
                             "objects and extract each one individually")
    parser.add_argument("--format", choices=['webm', 'png-seq'], default='webm',
                        help="Video output format: 'webm' (VP9+alpha) or 'png-seq' (PNG sequence)")
    parser.add_argument("--vram-optimize", action="store_true",
                        help="Run models in serial to save VRAM on lower end GPUs")
    parser.add_argument("--vram-limit", type=int, default=11,
                        help="VRAM limit in GB (default: 11)")

    args = parser.parse_args()

    # Collect input files
    files = get_media_files(args.input)
    if not files:
        print("Error: No media files found at '%s'" % args.input)
        sys.exit(1)

    # Detect video files
    video_files = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]

    print("=" * 60)
    print("  ROTOBOT — Automatic Rotoscoping")
    print("=" * 60)
    print("  Input:     %s (%d file%s)" % (
        args.input, len(files), "s" if len(files) != 1 else ""))
    if video_files:
        print("  Mode:      VIDEO (%d video%s)" % (
            len(video_files), "s" if len(video_files) != 1 else ""))
        print("  Format:    %s" % args.format)
    print("  Prompt:    %s" % args.prompt)
    print("  Threshold: %.2f (box) / %.2f (text)" % (
        args.threshold, args.text_threshold))
    print("  Refine:    %d" % args.refine)
    print("  Invert:    %s" % args.invert)
    print("  Auto-Inv:  %s" % args.auto_inventory)
    print("  VRAM Opt:  %s (Limit: %dGB)" % (args.vram_optimize, args.vram_limit))
    print("=" * 60)
    print()

    log.info("CLI started: input=%s, %d files (%d video, %d image), prompt='%s'",
        args.input, len(files), len(video_files), len(image_files), args.prompt)

    from rotobot_engine import RotobotEngine
    engine = RotobotEngine.get_instance()
    engine.set_vram_optimization(args.vram_optimize, args.vram_limit)

    # --- Video mode ---
    if video_files:
        from rotobot_engine import RotobotEngine
        engine = RotobotEngine.get_instance()

        out_dir = args.output or os.path.dirname(video_files[0])
        os.makedirs(out_dir, exist_ok=True)

        t_total = time.perf_counter()
        total_ok = 0

        for vid_path in video_files:
            basename = os.path.basename(vid_path)
            stem = os.path.splitext(basename)[0]

            info = engine.get_video_info(vid_path)
            print("  Video: %s (%dx%d, %.1f fps, %d frames, %.1fs)" % (
                basename, info['width'], info['height'],
                info['fps'], info['frames'], info['duration_s']))

            # Determine output path
            vid_format = args.format.replace('-', '_')  # 'png-seq' -> 'png_seq'
            if vid_format == 'png_seq':
                out_path = os.path.join(out_dir, stem + "_alpha_frames")
            else:
                out_path = os.path.join(out_dir, stem + "_alpha.webm")

            # Process frames
            def frame_gen():
                for idx, total_f, frame_rgb, alpha in engine.process_video_frames(
                    video_path=vid_path,
                    prompt=args.prompt,
                    box_threshold=args.threshold,
                    text_threshold=args.text_threshold,
                    refine=args.refine,
                    invert=args.invert,
                ):
                    print("    Frame %d/%d" % (idx + 1, total_f), end='\r')
                    yield (frame_rgb, alpha)

            ok = engine.save_video_rgba(
                video_path=vid_path,
                frame_generator=frame_gen(),
                output_path=out_path,
                fps=info['fps'],
                output_format=vid_format,
            )
            print()  # newline after \r
            if ok:
                print("  OK    %s -> %s" % (basename, os.path.basename(out_path)))
                total_ok += 1
            else:
                print("  FAIL  %s" % basename)

        total_time = time.perf_counter() - t_total
        print()
        print("=" * 60)
        print("  Video processing done! %d/%d succeeded in %.1fs" % (
            total_ok, len(video_files), total_time))
        print("=" * 60)
        log.info("Video processing finished: %d/%d in %.1fs", total_ok, len(video_files), total_time)

        # If there are also image files, fall through to process them
        if not image_files:
            return
        files = image_files
        print()
        print("  Also processing %d image files..." % len(image_files))

    # --- Auto-Inventory mode ---
    if args.auto_inventory:
        from rotobot_vision import VisionClient, auto_inventory

        # Determine output directory
        out_dir = args.output or os.path.dirname(files[0])
        os.makedirs(out_dir, exist_ok=True)

        client = VisionClient()
        print("Checking Vision API...")
        if not client.check_health():
            print("Launching Vision service...")
            if not client.launch_service():
                print("Error: Vision API unavailable. Start it manually.")
                sys.exit(1)

        # Load GSAM2 engine
        from rotobot_engine import RotobotEngine
        engine = RotobotEngine.get_instance()
        if not engine.ensure_models():
            print("Error: Failed to load AI models.")
            sys.exit(1)

        t_total = time.perf_counter()
        total_success = 0
        total_failed = 0

        for img_path in files:
            results = auto_inventory(
                image_path=img_path,
                output_dir=out_dir,
                box_threshold=args.threshold,
                text_threshold=args.text_threshold,
                refine=args.refine,
            )
            ok = sum(1 for _, _, s in results if s)
            total_success += ok
            if ok == 0:
                total_failed += 1

        total_time = time.perf_counter() - t_total
        print()
        print("=" * 60)
        print("  Auto-Inventory done! %d elements extracted in %.1fs" % (
            total_success, total_time))
        print("=" * 60)
        log.info("Auto-inventory finished: %d elements in %.1fs", total_success, total_time)
        return

    # --- Standard mode ---
    # Load engine
    from rotobot_engine import RotobotEngine
    engine = RotobotEngine.get_instance()

    if not engine.ensure_models():
        print("Error: Failed to load AI models. Check model paths.")
        sys.exit(1)

    # Process each file
    success = 0
    failed = 0
    t_total = time.perf_counter()

    # Optional tqdm progress bar
    try:
        from tqdm import tqdm
        file_iter = tqdm(files, desc="Processing", unit="img")
    except ImportError:
        file_iter = files

    for img_path in file_iter:
        basename = os.path.basename(img_path)

        # Determine output path
        if len(files) == 1 and args.output and not os.path.isdir(args.output):
            out_path = args.output
        else:
            out_path = make_output_path(img_path, args.output)

        t0 = time.perf_counter()

        alpha = engine.extract_alpha(
            image_path=img_path,
            prompt=args.prompt,
            box_threshold=args.threshold,
            text_threshold=args.text_threshold,
            refine=args.refine,
            invert=args.invert,
        )

        if alpha is None:
            print("  SKIP  %s - no objects detected" % basename)
            failed += 1
            continue

        if engine.save_rgba(img_path, alpha, out_path):
            elapsed = time.perf_counter() - t0
            print("  OK    %s -> %s (%.1fs)" % (basename, os.path.basename(out_path), elapsed))
            log.info("OK  %s -> %s (%.1fs)", basename, os.path.basename(out_path), elapsed)
            success += 1
        else:
            print("  FAIL  %s - save error" % basename)
            failed += 1

    total_time = time.perf_counter() - t_total
    print()
    print("=" * 60)
    print("  Done! %d/%d succeeded in %.1fs" % (success, len(files), total_time))
    if failed:
        print("  %d files had no detections or errors" % failed)
    print("=" * 60)
    log.info("CLI finished: %d/%d succeeded in %.1fs", success, len(files), total_time)


if __name__ == "__main__":
    main()
