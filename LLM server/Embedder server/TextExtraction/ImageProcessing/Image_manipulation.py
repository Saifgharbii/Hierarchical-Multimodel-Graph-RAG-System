from PIL import Image
from wand.image import Image as WandImage
from wand.exceptions import WandException
import io
import base64

# Supported MIME types
EXT_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "bmp": "image/bmp",
    "gif": "image/gif",
    "tiff": "image/tiff",
    "webp": "image/webp",
    "wmf": "image/x-wmf",  # Will be converted
    "emf": "image/x-emf",  # Will be converted
}


def encode_image_base64(image_bytes: bytes) -> str:
    """Encodes image bytes to Base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


def convert_emf_wmf_to_png(image_bytes: bytes, ext: str) -> tuple[bytes, str] | tuple[None, None]:
    """Uses wand (ImageMagick) to convert EMF/WMF to PNG."""
    try:
        with WandImage(blob=image_bytes) as img:
            img.format = 'png'
            converted_bytes = img.make_blob()
            return converted_bytes, "png"
    except WandException as e:
        print(f"[!] Wand conversion failed: {e}")
        return None, None


def convert_raster_to_png(image_bytes: bytes) -> tuple[bytes, str] | tuple[None, None]:
    """Uses Pillow to convert raster formats to PNG."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            return buf.getvalue(), "png"
    except Exception as e:
        print(f"[!] Pillow conversion failed: {e}")
        return None, None


def safe_convert_to_png(image_bytes: bytes, ext: str) -> tuple[bytes, str] | tuple[None, None]:
    """Converts image to PNG using appropriate method."""
    ext = ext.lower().lstrip(".")
    mime_type = EXT_TO_MIME.get(ext)
    if mime_type is None or mime_type.startswith("image/x-"):
        print(f"[i] Converting '{ext}' to PNG for compatibility...")
        if ext in ["emf", "wmf"]:
            try:
                return convert_emf_wmf_to_png(image_bytes, ext)
            except Exception as e:
                print(f"[!] Could not convert '{ext}' to png for this error {e}")
        else:
            try:
                return convert_raster_to_png(image_bytes)
            except Exception as e:
                print(f"[!] Could not convert '{ext}' to png for this error {e}")
    return None, None
