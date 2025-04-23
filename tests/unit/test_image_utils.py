import unittest
import os
import tempfile
import base64
from PIL import Image
from io import BytesIO
from ai_core.image_utils import (
    encode_image, validate_image, get_image_dimensions_from_base64
)

class TestImageUtils(unittest.TestCase):

    def setUp(self):
        # Create temporary files for testing
        self.test_dir = tempfile.TemporaryDirectory()

        # Create a valid small PNG file
        self.valid_png_path = os.path.join(self.test_dir.name, "valid.png")
        img_png = Image.new('RGB', (60, 30), color = 'red')
        img_png.save(self.valid_png_path, "PNG")

        # Create a valid small JPEG file
        self.valid_jpg_path = os.path.join(self.test_dir.name, "valid.jpg")
        img_jpg = Image.new('RGB', (50, 40), color = 'blue')
        img_jpg.save(self.valid_jpg_path, "JPEG")

        # Create a large file ( > max_size)
        self.large_file_path = os.path.join(self.test_dir.name, "large.bin")
        with open(self.large_file_path, "wb") as f:
            f.seek((21 * 1024 * 1024) - 1) # 21 MB - 1 byte
            f.write(b"\0") # Write a null byte to make file this size

        # Create a non-image file
        self.text_file_path = os.path.join(self.test_dir.name, "not_an_image.txt")
        with open(self.text_file_path, "w") as f:
            f.write("This is not an image.")

    def tearDown(self):
        # Clean up temporary directory
        self.test_dir.cleanup()

    def test_encode_image_png(self):
        encoded_data, media_type = encode_image(self.valid_png_path)
        self.assertEqual(media_type, "image/png")
        # Check if it's valid base64
        try:
            base64.b64decode(encoded_data)
        except Exception:
            self.fail("Encoded data is not valid base64")
        # Optional: decode and check magic bytes or size if needed

    def test_encode_image_jpeg(self):
        encoded_data, media_type = encode_image(self.valid_jpg_path)
        self.assertEqual(media_type, "image/jpeg") # Pillow saves as jpeg
        try:
            base64.b64decode(encoded_data)
        except Exception:
            self.fail("Encoded data is not valid base64")

    def test_encode_image_unsupported(self):
        with self.assertRaisesRegex(ValueError, "Unsupported image format"):
            encode_image(self.text_file_path)

    def test_validate_image_valid(self):
        try:
            validate_image(self.valid_png_path)
            validate_image(self.valid_jpg_path)
        except ValueError as e:
            self.fail(f"validate_image raised ValueError unexpectedly: {e}")

    def test_validate_image_not_found(self):
        with self.assertRaises(FileNotFoundError):
            validate_image("non_existent_file.png")

    def test_validate_image_too_large(self):
         # Use default max_size = 20MB
        with self.assertRaisesRegex(ValueError, "Image file too large"):
            validate_image(self.large_file_path)
        # Test with custom max_size
        with self.assertRaisesRegex(ValueError, "Image file too large"):
            validate_image(self.valid_png_path, max_size=10) # 10 bytes

    def test_validate_image_unsupported_format(self):
        with self.assertRaisesRegex(ValueError, "Unsupported image format"):
            validate_image(self.text_file_path)

    def test_get_image_dimensions_from_base64(self):
        # Encode the test PNG
        with open(self.valid_png_path, "rb") as f:
            png_data = f.read()
        b64_string = base64.b64encode(png_data).decode('utf-8')

        width, height = get_image_dimensions_from_base64(b64_string)
        self.assertEqual(width, 60)
        self.assertEqual(height, 30)

        # Test with MIME type prefix
        b64_string_with_prefix = f"data:image/png;base64,{b64_string}"
        width, height = get_image_dimensions_from_base64(b64_string_with_prefix)
        self.assertEqual(width, 60)
        self.assertEqual(height, 30)

if __name__ == '__main__':
    unittest.main()
