# --- top of file ---
try:
    import imghdr as _imghdr
except ModuleNotFoundError:        # Python ≥ 3.13
    _imghdr = None                 # We’ll emulate the tiny part we need
# -------------------------------------------------------
import base64
import os
from PIL import Image
from io import BytesIO
from typing import Tuple


def _detect_image_type_from_bytes(data: bytes) -> str | None:
    """Return the lowercase image format or None if undetectable."""
    try:
        with Image.open(BytesIO(data)) as img:
            return img.format.lower() if img.format else None
    except Exception:
        return None

def encode_image(image_path: str) -> Tuple[str, str]:
    with open(image_path, "rb") as image_file:
        file_content = image_file.read()

        # --- NEW: robust image‑type detection ---
        if _imghdr is not None:
            image_type = _imghdr.what(None, file_content)
        else:                                   # Python 3.13+
            image_type = _detect_image_type_from_bytes(file_content)
        # ---------------------------------------

        if image_type is None:
            raise ValueError(f"Unsupported image format for file: {image_path}")

        return base64.b64encode(file_content).decode('utf-8'), f"image/{image_type}"

def validate_image(image_path: str, max_size: int = 20 * 1024 * 1024) -> None:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if os.path.getsize(image_path) > max_size:
        raise ValueError(f"Image file too large: {image_path}")

    # --- NEW robust check ---
    if _imghdr is not None:
        image_type = _imghdr.what(image_path)
    else:
        with open(image_path, "rb") as f:
            image_type = _detect_image_type_from_bytes(f.read())
    # ------------------------

    if image_type is None:
        raise ValueError(f"Unsupported image format: {image_path}")

def get_image_dimensions_from_base64(base64_string):
    # Remove the MIME type prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    
    # Create a file-like object from the decoded data
    image_file = BytesIO(image_data)
    
    # Open the image using PIL
    with Image.open(image_file) as img:
        # Get the dimensions
        width, height = img.size
    
    return width, height 