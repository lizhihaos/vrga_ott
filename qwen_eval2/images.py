from io import BytesIO

from PIL import Image


def load_image(image_input):
    """Load any supported image input into a RGB PIL image."""

    if isinstance(image_input, dict) and "bytes" in image_input:
        return Image.open(BytesIO(image_input["bytes"])).convert("RGB")
    if isinstance(image_input, bytes):
        return Image.open(BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    return image_input

