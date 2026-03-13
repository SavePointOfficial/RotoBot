# Rotobot Engine
# Automatic rotoscoping via Grounded SAM2 (GroundingDINO + SAM2).
# Adapted from LabyrinthGameSandbox/labyrinth_modules/grounded_sam2_solver.py
#
# Takes a still image + text prompt (e.g. "person", "car . dog")
# and produces a production-ready alpha mask.

import os
import sys
import time
import threading
import numpy as np
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — resolve sandbox root for models and dependencies
# Priority: 1) config.json   2) ROTOBOT_SANDBOX_ROOT env var   3) parent dir
# ---------------------------------------------------------------------------
ROTOBOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_sandbox_root():
    """Find the root directory containing models/ (supports portable installs)."""
    import json as _json

    # 1. Check for config.json in the Rotobot directory
    config_path = os.path.join(ROTOBOT_DIR, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = _json.load(f)
            root = cfg.get('sandbox_root', '')
            if root:
                # Support relative paths (e.g. "." for portable installs)
                if not os.path.isabs(root):
                    root = os.path.normpath(os.path.join(ROTOBOT_DIR, root))
                if os.path.isdir(root):
                    return root
        except Exception:
            pass

    # 2. Check environment variable
    env_root = os.environ.get('ROTOBOT_SANDBOX_ROOT', '')
    if env_root and os.path.isdir(env_root):
        return env_root

    # 3. Fall back to ROTOBOT_DIR itself (portable standalone install)
    return ROTOBOT_DIR

SANDBOX_ROOT = _resolve_sandbox_root()

# Add sandbox root to sys.path so we can import groundingdino / sam2
if SANDBOX_ROOT not in sys.path:
    sys.path.insert(0, SANDBOX_ROOT)

# Also add the sam2 repo so `import sam2` works
SAM2_REPO_DIR = os.path.join(SANDBOX_ROOT, 'models', 'sam2', 'repo')
if SAM2_REPO_DIR not in sys.path:
    sys.path.insert(0, SAM2_REPO_DIR)

# Grounding DINO model files
GDINO_CONFIG = os.path.join(SANDBOX_ROOT, 'models', 'groundingdino',
                            'GroundingDINO_SwinT_OGC.py')
GDINO_WEIGHTS = os.path.join(SANDBOX_ROOT, 'models', 'groundingdino',
                             'groundingdino_swint_ogc.pth')

# SAM2 model files
SAM2_CONFIG_DIR = os.path.join(SAM2_REPO_DIR, 'sam2', 'configs')
SAM2_CHECKPOINT = os.path.join(SANDBOX_ROOT, 'models', 'sam2',
                               'sam2.1_hiera_tiny.pt')
SAM2_CONFIG = "sam2.1/sam2.1_hiera_t.yaml"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PROMPT = "foreground subject . main object"
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.20
DEFAULT_REFINE_STRENGTH = 1  # 0 = off, 1 = light, 2 = medium, 3 = heavy


# ---------------------------------------------------------------------------
# Video extensions
# ---------------------------------------------------------------------------
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
from rotobot_logging import get_logger
log = get_logger("ENGINE")


# ============================================================================
# Rotobot Engine
# ============================================================================

class RotobotEngine:
    """
    Loads GroundingDINO + SAM2 once, then extracts alpha masks from images.

    Usage:
        engine = RotobotEngine.get_instance()
        alpha = engine.extract_alpha("photo.jpg", prompt="person")
        # alpha is (H, W) uint8, 0 = transparent, 255 = opaque
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self._gdino_model = None
        self._sam2_predictor = None
        self._sam2_model = None       # raw SAM2 model (shared with auto mask gen)
        self._gdino_loaded = False
        self._sam2_loaded = False
        self._upscaler = None
        self._device = None
        self._model_lock = threading.Lock()
        self._vram_optimize = False
        self._vram_limit_gb = 11

    @property
    def is_loaded(self) -> bool:
        return self._gdino_loaded and self._sam2_loaded

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_gdino(self) -> bool:
        """Lazy-load Grounding DINO model."""
        if self._gdino_loaded:
            return True
        with self._model_lock:
            if self._gdino_loaded:
                return True
            try:
                log.info("Loading Grounding DINO...")
                t0 = time.perf_counter()

                import torch
                from groundingdino.util.inference import load_model

                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._gdino_model = load_model(GDINO_CONFIG, GDINO_WEIGHTS)

                elapsed = time.perf_counter() - t0
                self._gdino_loaded = True
                
                if self._vram_optimize:
                    self._gdino_model.to('cpu')
                    torch.cuda.empty_cache()
                    
                log.info("Grounding DINO loaded on %s in %.1fs", self._device, elapsed)
                return True
            except Exception as e:
                log.error("FAILED to load Grounding DINO: %s", e, exc_info=True)
                return False

    def _ensure_sam2(self) -> bool:
        """Lazy-load SAM2 image predictor."""
        if self._sam2_loaded:
            return True
        with self._model_lock:
            if self._sam2_loaded:
                return True
            try:
                log.info("Loading SAM2 image predictor...")
                t0 = time.perf_counter()

                import torch
                from hydra import compose, initialize_config_dir
                from hydra.core.global_hydra import GlobalHydra
                from omegaconf import OmegaConf
                from hydra.utils import instantiate
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                device = self._device or ('cuda' if torch.cuda.is_available() else 'cpu')
                self._device = device

                GlobalHydra.instance().clear()
                abs_config_dir = os.path.abspath(SAM2_CONFIG_DIR)
                with initialize_config_dir(config_dir=abs_config_dir, version_base=None):
                    cfg = compose(config_name=SAM2_CONFIG, overrides=[
                        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                    ])
                    OmegaConf.resolve(cfg)
                    model = instantiate(cfg.model, _recursive_=True)

                sd = torch.load(SAM2_CHECKPOINT, map_location="cpu",
                                weights_only=True)["model"]
                model.load_state_dict(sd)
                model = model.to(device)
                model.eval()

                self._sam2_model = model
                self._sam2_predictor = SAM2ImagePredictor(model)

                elapsed = time.perf_counter() - t0
                self._sam2_loaded = True
                
                if self._vram_optimize:
                    self._sam2_model.to('cpu')
                    torch.cuda.empty_cache()
                    
                log.info("SAM2 predictor loaded on %s in %.1fs", device, elapsed)
                return True
            except Exception as e:
                log.error("FAILED to load SAM2: %s", e, exc_info=True)
                return False

    def ensure_models(self) -> bool:
        """Pre-load both models. Returns True if successful."""
        return self._ensure_gdino() and self._ensure_sam2()

    def set_vram_optimization(self, enabled: bool, vram_gb: int = 11):
        """Enable memory-saving features for lower-VRAM GPUs."""
        self._vram_optimize = enabled
        self._vram_limit_gb = vram_gb
        if enabled:
            log.info("VRAM Optimization ENABLED (Limit: %d GB). Models will run in serial.", vram_gb)

    # ------------------------------------------------------------------
    # Automatic segmentation (no text prompts needed)
    # ------------------------------------------------------------------

    def segment_all(
        self,
        image_path: str,
        min_area: float = 0.005,
        max_area: float = 0.65,
        refine: int = DEFAULT_REFINE_STRENGTH,
        detail: int = 3,
    ) -> list:
        """
        Segment an image into all natural visual regions using SAM2's
        automatic mask generator. No text prompts needed.

        Args:
            image_path:  path to input image
            min_area:    minimum mask area as fraction of image (0-1)
            max_area:    maximum mask area as fraction of image (0-1)
            refine:      edge refinement strength
            detail:      1-5; 1=coarse (~10 regions), 5=fine (~100+ regions)

        Returns:
            List of (alpha, area_fraction, bbox) tuples sorted by area
            descending, where:
              alpha = (H, W) uint8 mask
              area_fraction = fraction of image covered (0-1)
              bbox = (x, y, w, h) in pixels
        """
        from PIL import Image

        if not self._ensure_sam2():
            return []

        try:
            pil_img = Image.open(image_path).convert("RGB")
            frame_rgb = np.array(pil_img)
        except Exception as e:
            log.error("Cannot load image %s: %s", image_path, e)
            return []

        return self.segment_all_from_array(frame_rgb, min_area, max_area, refine, detail)

    def segment_all_from_array(
        self,
        frame_rgb: np.ndarray,
        min_area: float = 0.005,
        max_area: float = 0.65,
        refine: int = DEFAULT_REFINE_STRENGTH,
        detail: int = 3,
    ) -> list:
        """
        Segment an RGB array into all natural visual regions.

        Args:
            detail: 1-5; 1=coarse (~10 regions), 5=fine (~100+ regions)

        Returns:
            List of (alpha, area_fraction, bbox) tuples
        """
        if not self._ensure_sam2():
            return []

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        if self._vram_optimize:
            import torch
            self._sam2_model.to(self._device)

        t0 = time.perf_counter()
        h, w = frame_rgb.shape[:2]
        total_pixels = h * w

        # Map detail (1-5) to SAM2 parameters
        detail = max(1, min(5, detail))
        # points_per_side: 16 → 24 → 32 → 48 → 64
        pts_map = {1: 16, 2: 24, 3: 32, 4: 48, 5: 64}
        # pred_iou_thresh: higher = fewer masks, lower = more
        iou_map = {1: 0.85, 2: 0.78, 3: 0.70, 4: 0.60, 5: 0.50}
        # stability_score_thresh: higher = fewer masks, lower = more
        stab_map = {1: 0.95, 2: 0.93, 3: 0.92, 4: 0.88, 5: 0.80}
        # Scale min_area down at higher detail to catch smaller elements
        min_area_effective = min_area / (1 + (detail - 1) * 0.3)

        pts = pts_map[detail]
        iou = iou_map[detail]
        stab = stab_map[detail]

        log.info("Running SAM2 auto-segmentation (detail=%d, pts=%d, iou=%.2f, stab=%.2f) on %dx%d...",
            detail, pts, iou, stab, w, h)

        mask_gen = SAM2AutomaticMaskGenerator(
            model=self._sam2_model,
            points_per_side=pts,
            points_per_batch=64,
            pred_iou_thresh=iou,
            stability_score_thresh=stab,
            crop_n_layers=1 if detail <= 3 else 2,
            min_mask_region_area=int(total_pixels * min_area_effective),
        )

        masks = mask_gen.generate(frame_rgb)
        t_seg = time.perf_counter() - t0

        if self._vram_optimize:
            import torch as _torch_mem
            self._sam2_model.to('cpu')
            _torch_mem.cuda.empty_cache()

        log.info("SAM2 auto-segmentation found %d raw masks in %.1fs", len(masks), t_seg)

        # Filter and sort
        results = []
        for i, m in enumerate(masks):
            seg = m["segmentation"]  # (H, W) bool
            area_frac = m["area"] / total_pixels
            bbox = m["bbox"]  # [x, y, w, h]

            if area_frac < min_area:
                log.debug("  mask[%d] area=%.2f%% — too small, skipping", i, area_frac * 100)
                continue
            if area_frac > max_area:
                log.debug("  mask[%d] area=%.2f%% — too large, skipping", i, area_frac * 100)
                continue

            alpha = (seg.astype(np.uint8) * 255)
            if refine > 0:
                alpha = self._refine_edges(alpha, strength=refine)

            log.debug("  mask[%d] area=%.1f%% bbox=[%d,%d,%d,%d] score=%.3f",
                i, area_frac * 100, bbox[0], bbox[1], bbox[2], bbox[3],
                m.get("predicted_iou", 0))

            results.append((alpha, area_frac, tuple(bbox)))

        # Sort by area descending (largest elements first)
        results.sort(key=lambda x: x[1], reverse=True)

        log.info("Auto-segmentation: %d masks passed filters (%.1f%% - %.1f%% area range)",
            len(results), min_area * 100, max_area * 100)

        return results

    def extract_alpha(
        self,
        image_path: str,
        prompt: str = DEFAULT_PROMPT,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        refine: int = DEFAULT_REFINE_STRENGTH,
        invert: bool = False,
        select_best: bool = False,
        max_coverage: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        Extract an alpha mask from an image file.

        Args:
            image_path:     path to input image (jpg, png, bmp, etc.)
            prompt:         GroundingDINO text prompt
            box_threshold:  detection confidence threshold (0-1)
            text_threshold: text matching threshold (0-1)
            refine:         edge refinement strength (0=off, 1=light, 3=heavy)
            invert:         if True, invert alpha (keep background)
            select_best:    if True, use only the highest-confidence detection
            max_coverage:   if >0, reject masks covering more than this fraction
                            of the image (e.g. 0.6 = reject if >60%% coverage)

        Returns:
            (H, W) uint8 numpy array or None
        """
        from PIL import Image

        try:
            pil_img = Image.open(image_path).convert("RGB")
            frame_rgb = np.array(pil_img)
        except Exception as e:
            log.error("Cannot load image %s: %s", image_path, e)
            return None

        return self.extract_alpha_from_array(
            frame_rgb, prompt, box_threshold, text_threshold, refine, invert,
            select_best=select_best, max_coverage=max_coverage,
        )

    def extract_alpha_from_array(
        self,
        frame_rgb: np.ndarray,
        prompt: str = DEFAULT_PROMPT,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        refine: int = DEFAULT_REFINE_STRENGTH,
        invert: bool = False,
        select_best: bool = False,
        max_coverage: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        Extract alpha mask from an RGB numpy array.

        Args:
            frame_rgb:     (H, W, 3) uint8 RGB numpy array
            select_best:   if True, use only the highest-confidence detection
            max_coverage:  if >0, reject masks covering more than this fraction
            (other args same as extract_alpha)

        Returns:
            (H, W) uint8 alpha or None
        """
        if not self._ensure_gdino():
            return None
        if not self._ensure_sam2():
            return None

        import torch
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T
        from PIL import Image

        h, w = frame_rgb.shape[:2]

        # --- GroundingDINO detection ---
        if self._vram_optimize:
            self._gdino_model.to(self._device)

        t0 = time.perf_counter()

        pil_image = Image.fromarray(frame_rgb)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(pil_image, None)

        boxes, logits, phrases = predict(
            model=self._gdino_model,
            image=image_transformed,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        t_detect = time.perf_counter() - t0

        if self._vram_optimize:
            import torch as _torch_mem
            self._gdino_model.to('cpu')
            _torch_mem.cuda.empty_cache()

        if len(boxes) == 0:
            log.info("No objects detected for prompt '%s' (%.2fs)", prompt, t_detect)
            return None

        # Log all detections with confidence scores
        for i, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
            cx_n, cy_n, bw_n, bh_n = box.tolist()
            box_area = bw_n * bh_n  # normalized area (0-1)
            log.debug("  det[%d] '%s' conf=%.3f area=%.1f%% box=[%.2f,%.2f,%.2f,%.2f]",
                i, phrase.strip(), score.item(), box_area * 100,
                cx_n, cy_n, bw_n, bh_n)

        # --- select_best: keep only the highest-confidence detection ---
        if select_best and len(boxes) > 1:
            best_idx = logits.argmax().item()
            log.info("select_best: keeping det[%d] '%s' (conf=%.3f) out of %d detections",
                best_idx, phrases[best_idx].strip(), logits[best_idx].item(), len(boxes))
            boxes = boxes[best_idx:best_idx+1]
            logits = logits[best_idx:best_idx+1]
            phrases = [phrases[best_idx]]

        # --- Filter out oversized boxes (likely scene-level detections) ---
        if select_best:
            keep = []
            for i, box in enumerate(boxes):
                cx_n, cy_n, bw_n, bh_n = box.tolist()
                area = bw_n * bh_n
                if area > 0.70:
                    log.warning("Dropping det[%d] '%s' — box covers %.0f%% of image (too large)",
                        i, phrases[i].strip(), area * 100)
                else:
                    keep.append(i)
            if not keep:
                log.info("All detections for '%s' were oversized — skipping", prompt)
                return None
            if len(keep) < len(boxes):
                import torch as _torch
                boxes = boxes[keep]
                logits = logits[_torch.tensor(keep)]
                phrases = [phrases[i] for i in keep]

        log.info("Detected %d objects in %.2fs: %s",
            len(boxes), t_detect,
            ", ".join(p.strip() for p in phrases)
        )

        # --- Convert boxes to pixel coords ---
        boxes_xyxy = []
        for box in boxes:
            cx_n, cy_n, bw_n, bh_n = box.tolist()
            x1 = (cx_n - bw_n / 2) * w
            y1 = (cy_n - bh_n / 2) * h
            x2 = (cx_n + bw_n / 2) * w
            y2 = (cy_n + bh_n / 2) * h
            boxes_xyxy.append([x1, y1, x2, y2])

        boxes_np = np.array(boxes_xyxy, dtype=np.float32)

        # --- SAM2 segmentation ---
        if self._vram_optimize:
            self._sam2_model.to(self._device)

        t1 = time.perf_counter()

        self._sam2_predictor.set_image(frame_rgb)

        input_boxes = torch.tensor(boxes_np, device=self._device)

        masks_out, scores_out, _ = self._sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        t_segment = time.perf_counter() - t1

        if self._vram_optimize:
            import torch as _torch_mem
            self._sam2_model.to('cpu')
            _torch_mem.cuda.empty_cache()

        # masks_out shape: (N, 1, H, W) or (N, H, W)
        if masks_out.ndim == 4:
            masks_out = masks_out[:, 0]  # (N, H, W)

        log.debug("Segmented %d masks in %.2fs", len(masks_out), t_segment)

        # --- Union all masks into one alpha ---
        combined = np.zeros((h, w), dtype=bool)
        for i in range(len(masks_out)):
            mask_i = masks_out[i].astype(bool)
            mask_coverage = 100.0 * np.count_nonzero(mask_i) / (h * w)
            log.debug("  mask[%d] coverage=%.1f%%", i, mask_coverage)
            combined |= mask_i

        # Convert to uint8 alpha
        alpha = (combined.astype(np.uint8) * 255)

        # --- Edge refinement ---
        if refine > 0:
            alpha = self._refine_edges(alpha, strength=refine)

        # --- Invert if requested ---
        if invert:
            alpha = 255 - alpha

        coverage = 100.0 * np.count_nonzero(alpha > 127) / (h * w)
        log.debug("Alpha extracted: %dx%d, coverage %.1f%%", w, h, coverage)

        # --- Reject masks that cover too much of the image ---
        if max_coverage > 0 and coverage > max_coverage * 100.0:
            log.warning("Rejecting mask for '%s' — coverage %.1f%% exceeds max %.0f%%",
                prompt, coverage, max_coverage * 100.0)
            return None

        # Warn if coverage is suspiciously high
        if coverage > 90.0:
            log.warning("Very high coverage (%.1f%%) for prompt '%s'",
                coverage, prompt)

        return alpha

    # ------------------------------------------------------------------
    # Edge refinement
    # ------------------------------------------------------------------

    @staticmethod
    def _refine_edges(alpha: np.ndarray, strength: int = 1) -> np.ndarray:
        """
        Smooth jagged mask edges for production-quality alpha.

        strength 1: light — 1px morphological close + small Gaussian
        strength 2: medium — 2px close + medium Gaussian
        strength 3: heavy — 3px close + large Gaussian + slight dilate
        """
        import cv2

        kernel_size = max(1, strength) * 2 + 1  # 3, 5, 7
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # Morphological close to fill small holes
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        # Gaussian blur for smooth edges
        blur_size = max(1, strength) * 2 + 1  # 3, 5, 7
        alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)

        # Re-threshold to keep alpha crisp but with soft edges
        # Use a softer threshold for heavier refinement
        if strength >= 3:
            # Keep semi-transparent edge pixels
            pass  # Leave the Gaussian blur result as-is for softest edges
        else:
            # Threshold to mostly binary with slight edge softness
            _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            # One more light blur for sub-pixel smoothness
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

        return alpha

    # ------------------------------------------------------------------
    # Save RGBA PNG
    # ------------------------------------------------------------------

    @staticmethod
    def save_rgba(image_path: str, alpha: np.ndarray,
                  output_path: str, feather: int = 0,
                  crop: bool = False, upscale_to: int = 0) -> bool:
        """
        Combine original image with alpha mask and save as RGBA PNG.

        Args:
            image_path:   path to original image
            alpha:        (H, W) uint8 alpha mask
            output_path:  where to save the RGBA PNG
            feather:      edge feathering in pixels (0 = off)
            crop:         if True, crop output to bounding box of alpha > 0
            upscale_to:   if > 0, upscale so longest side = this many px
                          (only upscales, never downscales)

        Returns:
            True on success
        """
        from PIL import Image
        import cv2

        try:
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img)

            h, w = img_array.shape[:2]
            ah, aw = alpha.shape[:2]

            # Resize alpha to match image if needed
            if (ah, aw) != (h, w):
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

            # --- Feathering: blur the edge zone of the alpha ---
            if feather > 0:
                k = feather * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                dilated = cv2.dilate(alpha, kernel)
                eroded = cv2.erode(alpha, kernel)
                edge_zone = (dilated > 0) & (eroded < 255)
                blurred = cv2.GaussianBlur(alpha, (k, k), feather * 0.5)
                alpha = np.where(edge_zone, blurred, alpha)
                log.debug("Feathered edges: %dpx", feather)

            # Build RGBA
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_array
            rgba[:, :, 3] = alpha

            out_img = Image.fromarray(rgba, 'RGBA')

            # --- Crop to content ---
            if crop:
                bbox = out_img.getbbox()  # returns (left, top, right, bottom) of non-zero alpha
                if bbox is not None:
                    # Add small padding
                    pad = 2
                    left = max(0, bbox[0] - pad)
                    top = max(0, bbox[1] - pad)
                    right = min(w, bbox[2] + pad)
                    bottom = min(h, bbox[3] + pad)
                    out_img = out_img.crop((left, top, right, bottom))
                    cw, ch = out_img.size
                    log.debug("Cropped: %dx%d -> %dx%d", w, h, cw, ch)

            # --- Upscale if small (Real-ESRGAN) ---
            if upscale_to > 0:
                cw, ch = out_img.size
                longest = max(cw, ch)
                if longest < upscale_to:
                    out_img = RotobotEngine._upscale_rgba(out_img, upscale_to)

            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            out_img.save(output_path, 'PNG')

            log.info("Saved: %s (%dx%d)", output_path, out_img.size[0], out_img.size[1])
            return True
        except Exception as e:
            log.error("Failed to save %s: %s", output_path, e, exc_info=True)
            return False
    # ------------------------------------------------------------------
    # Real-ESRGAN upscaling
    # ------------------------------------------------------------------

    _esrgan_upscaler = None

    @classmethod
    def _get_upscaler(cls):
        """Lazy-load Real-ESRGAN 4x upscaler (downloads weights on first use)."""
        if cls._esrgan_upscaler is not None:
            return cls._esrgan_upscaler

        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )

            # Model weights will be auto-downloaded by RealESRGANer
            model_path = os.path.join(
                os.path.dirname(__file__), 'models', 'RealESRGAN_x4plus.pth')

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            upscaler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=0,         # 0 = no tiling (for small extracted elements)
                tile_pad=10,
                pre_pad=0,
                half=False,     # use fp32 for CPU compatibility
                device=device,
            )

            cls._esrgan_upscaler = upscaler
            log.info("Real-ESRGAN upscaler loaded on %s", device)
            return upscaler

        except Exception as e:
            log.warning("Could not load Real-ESRGAN: %s (will fall back to LANCZOS)", e)
            return None

    @staticmethod
    def _upscale_rgba(pil_img, target_longest: int):
        """
        Upscale an RGBA PIL image using Real-ESRGAN for RGB
        and LANCZOS for the alpha channel.

        Falls back to LANCZOS for everything if Real-ESRGAN unavailable.
        """
        from PIL import Image

        cw, ch = pil_img.size
        longest = max(cw, ch)

        upscaler = RotobotEngine._get_upscaler()

        if upscaler is not None:
            try:
                # Split into RGB + Alpha
                rgba_arr = np.array(pil_img)
                rgb_arr = rgba_arr[:, :, :3]
                alpha_arr = rgba_arr[:, :, 3]

                # Real-ESRGAN expects BGR
                import cv2
                bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

                # Upscale RGB with Real-ESRGAN (4x)
                output_bgr, _ = upscaler.enhance(bgr, outscale=4)
                output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

                # Upscale alpha with LANCZOS (neural upscale adds artifacts to masks)
                oh, ow = output_rgb.shape[:2]
                alpha_pil = Image.fromarray(alpha_arr, 'L')
                alpha_up = alpha_pil.resize((ow, oh), Image.LANCZOS)

                # Recombine
                out_rgba = np.zeros((oh, ow, 4), dtype=np.uint8)
                out_rgba[:, :, :3] = output_rgb
                out_rgba[:, :, 3] = np.array(alpha_up)
                result = Image.fromarray(out_rgba, 'RGBA')

                # Resize to target (Real-ESRGAN does 4x, we may want different)
                rw, rh = result.size
                r_longest = max(rw, rh)
                if r_longest != target_longest:
                    scale = target_longest / r_longest
                    final_w = int(rw * scale)
                    final_h = int(rh * scale)
                    result = result.resize((final_w, final_h), Image.LANCZOS)

                log.debug("Real-ESRGAN upscaled: %dx%d -> %dx%d",
                    cw, ch, result.size[0], result.size[1])
                return result

            except Exception as e:
                log.warning("Real-ESRGAN failed, falling back to LANCZOS: %s", e)

        # Fallback: LANCZOS
        scale = target_longest / longest
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        result = pil_img.resize((new_w, new_h), Image.LANCZOS)
        log.debug("LANCZOS upscaled: %dx%d -> %dx%d", cw, ch, new_w, new_h)
        return result

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------

    def process_video_frames(
        self,
        video_path: str,
        prompt: str = DEFAULT_PROMPT,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        refine: int = DEFAULT_REFINE_STRENGTH,
        invert: bool = False,
        color_key: bool = False,
        key_color: tuple = (0, 0, 0),
        key_tolerance: float = 30.0,
        feather: int = 0,
        on_progress=None,
        cancel_event=None,
    ):
        """
        Generator: decode video frame-by-frame and yield alpha masks.

        Yields:
            (frame_idx, total_frames, frame_rgb, alpha) for each frame.
            alpha is (H, W) uint8 — 0=transparent, 255=opaque.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error("Cannot open video: %s", video_path)
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Video opened: %s — %d frames, %.1f fps, %dx%d",
                 video_path, total, fps, w, h)

        # Pre-load models once (not per frame)
        if not color_key:
            self.ensure_models()

        idx = 0
        while True:
            if cancel_event and cancel_event.is_set():
                log.info("Video processing cancelled at frame %d/%d", idx, total)
                break

            ret, bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if color_key:
                alpha = self.extract_alpha_from_color_array(
                    frame_rgb, key_color, key_tolerance, refine, invert)
            else:
                alpha = self.extract_alpha_from_array(
                    frame_rgb, prompt, box_threshold, text_threshold,
                    refine, invert)

            if alpha is None:
                # No detection — fully transparent frame
                alpha = np.zeros((h, w), dtype=np.uint8)

            # Apply feathering
            if feather > 0:
                k = feather * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                dilated = cv2.dilate(alpha, kernel)
                eroded = cv2.erode(alpha, kernel)
                edge_zone = (dilated > 0) & (eroded < 255)
                blurred = cv2.GaussianBlur(alpha, (k, k), feather * 0.5)
                alpha = np.where(edge_zone, blurred, alpha)

            if on_progress:
                on_progress(idx, total)

            yield (idx, total, frame_rgb, alpha)
            idx += 1

        cap.release()
        log.info("Video processing complete: %d frames", idx)

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get basic video metadata.

        Returns dict with keys: frames, fps, width, height, duration_s
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        info = {
            'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS) or 30.0,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info['duration_s'] = info['frames'] / info['fps'] if info['fps'] else 0
        cap.release()
        return info

    @staticmethod
    def save_video_rgba(
        video_path: str,
        frame_generator,
        output_path: str,
        fps: float = 30.0,
        output_format: str = 'webm',
    ) -> bool:
        """
        Encode RGBA video from a generator of (frame_rgb, alpha) tuples.

        Args:
            video_path:       original video (for audio copy, fps fallback)
            frame_generator:  iterable of (frame_rgb, alpha) tuples
            output_path:      where to save
            fps:              frames per second
            output_format:    'webm' for VP9+alpha, 'png_seq' for PNG sequence

        Returns:
            True on success
        """
        import cv2
        import subprocess
        import shutil
        import tempfile

        if output_format == 'png_seq':
            # --- PNG image sequence ---
            seq_dir = output_path  # treat output_path as directory
            os.makedirs(seq_dir, exist_ok=True)

            count = 0
            for frame_rgb, alpha in frame_generator:
                from PIL import Image
                h, w = frame_rgb.shape[:2]
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[:, :, :3] = frame_rgb
                rgba[:, :, 3] = alpha
                out_img = Image.fromarray(rgba, 'RGBA')
                out_img.save(os.path.join(seq_dir, "frame_%06d.png" % count), 'PNG')
                count += 1

            log.info("PNG sequence saved: %d frames to %s", count, seq_dir)
            return count > 0

        # --- WebM VP9 + alpha ---
        # Check for ffmpeg
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            log.error("FFmpeg not found on PATH — cannot encode alpha video")
            return False

        # Write raw RGBA frames to a temp file, then encode with ffmpeg
        tmp_dir = tempfile.mkdtemp(prefix='rotobot_vid_')
        raw_path = os.path.join(tmp_dir, 'rgba_raw.bin')

        try:
            width = height = 0
            count = 0

            with open(raw_path, 'wb') as raw_f:
                for frame_rgb, alpha in frame_generator:
                    h, w = frame_rgb.shape[:2]
                    if width == 0:
                        width, height = w, h

                    rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    rgba[:, :, :3] = frame_rgb
                    rgba[:, :, 3] = alpha
                    raw_f.write(rgba.tobytes())
                    count += 1

            if count == 0:
                log.error("No frames to encode")
                return False

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

            # Encode with FFmpeg: rawvideo RGBA -> VP9 + alpha WebM
            cmd = [
                ffmpeg_path, '-y',
                '-f', 'rawvideo',
                '-pixel_format', 'rgba',
                '-video_size', '%dx%d' % (width, height),
                '-framerate', '%.3f' % fps,
                '-i', raw_path,
                '-c:v', 'libvpx-vp9',
                '-pix_fmt', 'yuva420p',
                '-b:v', '2M',
                '-auto-alt-ref', '0',
                output_path,
            ]

            log.info("Encoding WebM: %d frames at %.1f fps, %dx%d",
                     count, fps, width, height)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                log.error("FFmpeg failed: %s", result.stderr[-500:] if result.stderr else "unknown")
                return False

            log.info("WebM saved: %s", output_path)
            return True

        except Exception as e:
            log.error("Video encoding failed: %s", e, exc_info=True)
            return False
        finally:
            # Clean up temp files
            try:
                import shutil as _sh
                _sh.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def extract_alpha_from_color(
        self,
        image_path: str,
        key_color: tuple = (0, 0, 0),
        tolerance: float = 30.0,
        refine: int = DEFAULT_REFINE_STRENGTH,
        invert: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Extract alpha by keying out a specific background color.

        No AI models needed — this is pure color-distance keying,
        perfect for images on black, green screen, or any solid bg.

        Args:
            image_path:  path to input image
            key_color:   (R, G, B) color to remove (0-255 each)
            tolerance:   how far from key_color counts as background
                         (in CIE LAB space, ~10=tight, ~50=loose)
            refine:      edge refinement strength (0=off, 1=light, 3=heavy)
            invert:      if True, invert alpha

        Returns:
            (H, W) uint8 alpha — 0=transparent, 255=opaque — or None
        """
        from PIL import Image

        try:
            pil_img = Image.open(image_path).convert("RGB")
            frame_rgb = np.array(pil_img)
        except Exception as e:
            log.error("Cannot load image %s: %s", image_path, e)
            return None

        return self.extract_alpha_from_color_array(
            frame_rgb, key_color, tolerance, refine, invert
        )

    def extract_alpha_from_color_array(
        self,
        frame_rgb: np.ndarray,
        key_color: tuple = (0, 0, 0),
        tolerance: float = 30.0,
        refine: int = DEFAULT_REFINE_STRENGTH,
        invert: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Extract alpha from RGB array by keying out a background color.

        Uses direct RGB Euclidean distance for precise color separation,
        especially effective for pure black/green/blue screen backgrounds.
        """
        import cv2

        t0 = time.perf_counter()

        h, w = frame_rgb.shape[:2]

        # Compute per-pixel RGB Euclidean distance from key color
        frame_f = frame_rgb.astype(np.float32)
        key = np.array(key_color, dtype=np.float32).reshape(1, 1, 3)
        diff = frame_f - key
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        # Hard cutoff: pixels within 40% of tolerance are fully transparent.
        # Soft ramp: pixels between 40% and 100% of tolerance get partial alpha.
        # Beyond tolerance: fully opaque.
        inner = tolerance * 0.4
        outer = tolerance

        alpha = np.clip((dist - inner) / max(outer - inner, 0.01), 0.0, 1.0)
        alpha = (alpha * 255).astype(np.uint8)

        # Edge refinement
        if refine > 0:
            alpha = self._refine_edges(alpha, strength=refine)

        # Invert if requested
        if invert:
            alpha = 255 - alpha

        elapsed = time.perf_counter() - t0
        log.info("Color key: (%d,%d,%d) tol=%.0f -> %dx%d, coverage %.1f%% (%.2fs)",
            key_color[0], key_color[1], key_color[2], tolerance,
            w, h, 100.0 * np.count_nonzero(alpha > 127) / (h * w), elapsed
        )

        return alpha
