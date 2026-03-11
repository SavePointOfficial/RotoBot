# Rotobot Vision — Auto-Inventory via LLaMA Vision API
#
# Sends images to a local LLaMA Vision endpoint to detect all distinct
# objects, then extracts each one through the existing GSAM2 pipeline.
#
# Usage:
#     from rotobot_vision import VisionClient, auto_inventory
#     elements = VisionClient().analyze_image("table.jpg")
#     # ["skull ornament", "chrome bumper", "neon tube", ...]
#     auto_inventory("table.jpg", "./output/")

import os
import sys
import json
import re
import time
import base64
import subprocess
from typing import List, Optional, Tuple

ROTOBOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROTOBOT_DIR not in sys.path:
    sys.path.insert(0, ROTOBOT_DIR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VISION_API_URL = "http://localhost:5001"
VISION_SERVICE_DIR = r"C:\Save Point V7\SPGRADER\ai_services\vision_llama_service"
VISION_LAUNCHER = os.path.join(VISION_SERVICE_DIR, "launch_absolute.bat")

# Prompt engineered to return clean JSON array of physical objects
INVENTORY_PROMPT = (
    "You are an object inventory system. Analyze this image and list every "
    "distinct, individual physical object or element visible. "
    "Return ONLY a JSON array of short descriptive strings. Rules:\n"
    "- Each string should be 1-3 words describing one specific object\n"
    "- Include materials/style when distinctive (e.g. \"rusted pipe\" not just \"pipe\")\n"
    "- Do NOT include abstract concepts (lighting, shadows, atmosphere, background)\n"
    "- Do NOT include surfaces (floor, wall, ceiling) unless they are decorative elements\n"
    "- Aim for completeness - list everything you can identify\n"
    "- Output ONLY the JSON array, nothing else\n"
    'Example: ["skull ornament", "chrome bumper", "neon tube", "gothic pillar"]'
)

API_TIMEOUT = 120  # seconds — vision models can be slow
API_RETRIES = 2


from rotobot_logging import get_logger
log = get_logger("VISION")


# ============================================================================
# Vision API Client
# ============================================================================

class VisionClient:
    """
    Client for the local LLaMA Vision API (Unsloth, port 5001).
    Sends images and gets back structured object inventories.
    """

    def __init__(self, base_url: str = VISION_API_URL, timeout: int = API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def check_health(self) -> bool:
        """Check if the Vision API is running and model is loaded."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(self.base_url + "/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                loaded = data.get("model_status", "").lower() == "loaded"
                if loaded:
                    log.info("Vision API healthy -- model: %s", data.get("model_name_or_path", "?"))
                else:
                    log.info("Vision API reachable but model not loaded")
                return loaded
        except urllib.error.URLError:
            log.debug("Vision API not reachable at %s", self.base_url)
            return False
        except Exception as e:
            log.error("Vision API health check error: %s", e)
            return False

    # ------------------------------------------------------------------
    # Launch service
    # ------------------------------------------------------------------

    def launch_service(self) -> bool:
        """
        Launch the Vision API service in background if not running.
        Returns True if service was launched or already running.
        """
        if self.check_health():
            return True

        if not os.path.exists(VISION_LAUNCHER):
            log.error("Vision service launcher not found: %s", VISION_LAUNCHER)
            return False

        log.info("Launching Vision service from %s...", VISION_LAUNCHER)
        try:
            # Launch in a new console window, don't wait
            subprocess.Popen(
                ["cmd", "/c", "start", "Vision Service", VISION_LAUNCHER],
                cwd=VISION_SERVICE_DIR,
                shell=True,
            )

            # Wait for it to come up (model loading takes time)
            log.info("Waiting for Vision service to start (this may take 30-60s)...")
            for i in range(60):  # Up to 60 seconds
                time.sleep(1)
                if self.check_health():
                    log.info("Vision service is ready!")
                    return True
                if i % 10 == 9:
                    log.info("Still waiting... (%ds)", i + 1)

            log.error("Vision service did not start within 60 seconds")
            return False
        except Exception as e:
            log.error("Failed to launch Vision service: %s", e, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Analyze image → JSON list of objects
    # ------------------------------------------------------------------

    def analyze_image(
        self,
        image_path: str,
        prompt: str = INVENTORY_PROMPT,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[str]:
        """
        Send an image to the Vision API and get a list of detected objects.

        Args:
            image_path:  path to image file
            prompt:      the system/user prompt for the LLM
            max_tokens:  max response length
            temperature: lower = more deterministic

        Returns:
            List of object name strings, e.g. ["skull", "chrome bumper", ...]
            Returns empty list on failure.
        """
        import urllib.request
        import urllib.error

        # Encode image as base64
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            log.error("Cannot read image %s: %s", image_path, e)
            return []

        # Build request
        payload = json.dumps({
            "image_base64": image_b64,
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.base_url + "/analyze",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # Send with retries
        for attempt in range(API_RETRIES + 1):
            try:
                log.info("Analyzing image: %s (attempt %d)...",
                    os.path.basename(image_path), attempt + 1)
                t0 = time.perf_counter()

                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())

                elapsed = time.perf_counter() - t0
                raw_text = data.get("analysis_result", "")

                log.info("Vision API responded in %.1fs", elapsed)

                # Parse the JSON array from the response
                elements = self._parse_json_array(raw_text)

                if elements:
                    log.info("Found %d elements: %s",
                        len(elements),
                        ", ".join(elements[:5]) + ("..." if len(elements) > 5 else "")
                    )
                    return elements
                else:
                    log.warning("Could not parse elements from response: %s", raw_text[:200])
                    if attempt < API_RETRIES:
                        log.info("Retrying...")
                        continue
                    return []

            except urllib.error.URLError as e:
                log.error("Vision API request failed: %s", e)
                if attempt < API_RETRIES:
                    time.sleep(2)
                    continue
                return []
            except Exception as e:
                log.error("Vision API error: %s", e, exc_info=True)
                if attempt < API_RETRIES:
                    time.sleep(2)
                    continue
                return []

        return []

    # ------------------------------------------------------------------
    # JSON parsing (robust — handles LLM quirks)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_array(text: str) -> List[str]:
        """
        Extract a JSON array of strings from LLM output.
        Handles common LLM quirks: markdown code fences, preamble text, etc.
        """
        text = text.strip()

        # Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(x).strip() for x in result if str(x).strip()]
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from markdown code block
        # ```json\n[...]\n```  or  ```\n[...]\n```
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_block:
            try:
                result = json.loads(code_block.group(1).strip())
                if isinstance(result, list):
                    return [str(x).strip() for x in result if str(x).strip()]
            except json.JSONDecodeError:
                pass

        # Try to find a JSON array anywhere in the text
        bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
        if bracket_match:
            try:
                result = json.loads(bracket_match.group(0))
                if isinstance(result, list):
                    return [str(x).strip() for x in result if str(x).strip()]
            except json.JSONDecodeError:
                pass

        # Last resort: try to extract quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return [q.strip() for q in quoted if q.strip()]

        return []


# ============================================================================
# Auto-Inventory: analyze + extract all elements
# ============================================================================

def auto_inventory(
    image_path: str,
    output_dir: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    refine: int = 1,
    vision_url: str = VISION_API_URL,
    progress_callback=None,
) -> List[Tuple[str, str, bool]]:
    """
    Full auto-inventory pipeline for a single image:
    1. Send to Vision API → get list of elements
    2. For each element, run GSAM2 extraction
    3. Save individual RGBA PNGs

    Args:
        image_path:        path to input image
        output_dir:        where to save output PNGs
        box_threshold:     GSAM2 detection threshold
        text_threshold:    GSAM2 text threshold
        refine:            edge refinement strength
        vision_url:        Vision API base URL
        progress_callback: optional fn(current, total, element_name) for progress

    Returns:
        List of (element_name, output_path, success) tuples
    """
    from rotobot_engine import RotobotEngine

    basename = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get element inventory from Vision API
    client = VisionClient(base_url=vision_url)
    elements = client.analyze_image(image_path)

    if not elements:
        log.info("No elements detected in %s", os.path.basename(image_path))
        return []

    log.info("Inventoried %d elements in %s", len(elements), os.path.basename(image_path))

    # Step 2: Load GSAM2 engine
    engine = RotobotEngine.get_instance()
    if not engine.ensure_models():
        log.error("Failed to load GSAM2 models")
        return []

    # Step 3: Extract each element
    results = []
    for i, element in enumerate(elements):
        if progress_callback:
            progress_callback(i + 1, len(elements), element)

        # Sanitize element name for filename
        safe_name = re.sub(r'[^\w\s-]', '', element).strip()
        safe_name = re.sub(r'[\s]+', '_', safe_name).lower()
        if not safe_name:
            safe_name = "element_%d" % (i + 1)

        out_path = os.path.join(output_dir, "%s_%s_alpha.png" % (basename, safe_name))

        log.info("  [%d/%d] Extracting: %s", i + 1, len(elements), element)
        t0 = time.perf_counter()

        alpha = engine.extract_alpha(
            image_path=image_path,
            prompt=element,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            refine=refine,
            invert=False,
            select_best=True,
            max_coverage=0.65,
        )

        if alpha is not None:
            success = engine.save_rgba(image_path, alpha, out_path)
            elapsed = time.perf_counter() - t0
            if success:
                log.info("    OK -> %s (%.1fs)", os.path.basename(out_path), elapsed)
            results.append((element, out_path, success))
        else:
            log.info("    SKIP -- no detection for '%s'", element)
            results.append((element, "", False))

    # Summary
    ok = sum(1 for _, _, s in results if s)
    log.info("Auto-inventory complete: %d/%d elements extracted from %s",
        ok, len(elements), os.path.basename(image_path))

    return results


def auto_inventory_batch(
    input_dir: str,
    output_dir: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    refine: int = 1,
    vision_url: str = VISION_API_URL,
    image_callback=None,
    element_callback=None,
) -> dict:
    """
    Auto-inventory all images in a directory.

    Args:
        input_dir:        folder containing images
        output_dir:       base output folder (subfolders created per image)
        box_threshold:    GSAM2 detection threshold
        text_threshold:   GSAM2 text threshold
        refine:           edge refinement strength
        vision_url:       Vision API base URL
        image_callback:   fn(img_index, total_images, filename) per image
        element_callback: fn(elem_index, total_elements, element_name) per element

    Returns:
        Summary dict with counts
    """
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]

    if not files:
        log.info("No images found in %s", input_dir)
        return {"images": 0, "elements": 0, "extracted": 0}

    total_elements = 0
    total_extracted = 0

    for img_i, fpath in enumerate(files):
        if image_callback:
            image_callback(img_i + 1, len(files), os.path.basename(fpath))

        results = auto_inventory(
            image_path=fpath,
            output_dir=output_dir,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            refine=refine,
            vision_url=vision_url,
            progress_callback=element_callback,
        )

        total_elements += len(results)
        total_extracted += sum(1 for _, _, s in results if s)

    log.info("Batch complete: %d images, %d elements found, %d extracted",
        len(files), total_elements, total_extracted)

    return {
        "images": len(files),
        "elements": total_elements,
        "extracted": total_extracted,
    }
