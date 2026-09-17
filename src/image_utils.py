import cv2
import os


def load_image(image_path):
    """
    Load an image from the specified path.
    Returns the image if successful, otherwise raises an error.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read the image: {image_path}"
        )

    return image


def get_image_info(image):
    """
    Display basic information about an image.
    """

    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    print("\n========== IMAGE INFORMATION ==========")
    print(f"Width       : {width} pixels")
    print(f"Height      : {height} pixels")
    print(f"Channels    : {channels}")
    print(f"Image Shape : {image.shape}")
    print("========================================")


def save_image(image, output_path):
    """
    Save a processed image to the specified path.
    """

    folder = os.path.dirname(output_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    success = cv2.imwrite(output_path, image)

    if not success:
        raise IOError(
            f"Unable to save image: {output_path}"
        )

    print(f"Image saved successfully: {output_path}")