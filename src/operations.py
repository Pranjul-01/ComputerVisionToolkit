import cv2
import matplotlib.pyplot as plt

def save_histogram(image, output_path):
    """
    Generate and save a grayscale image histogram.
    """

    if len(image.shape) != 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(histogram)
    plt.title("Grayscale Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Histogram saved successfully: {output_path}")

def convert_to_grayscale(image):
    """
    Convert a BGR color image into grayscale.
    """

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return gray_image


def resize_image(image, width=640, height=360):
    """
    Resize an image to the specified width and height.
    """

    resized_image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    return resized_image


def apply_gaussian_filter(image, kernel_size=5):
    """
    Apply Gaussian blur to reduce image noise.
    """

    if kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size must be an odd number."
        )

    blurred_image = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

    return blurred_image

def calculate_histogram(image):
    """
    Calculate the grayscale image histogram.
    """

    if len(image.shape) != 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    return histogram


def equalize_histogram(image):
    """
    Improve grayscale image contrast using
    histogram equalization.
    """

    if len(image.shape) != 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    equalized_image = cv2.equalizeHist(image)

    return equalized_image

def sobel_edge_detection(image):
    """
    Detect edges using the Sobel operator.
    Returns the gradient magnitude image.
    """

    if len(image.shape) != 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    sobel_x = cv2.Sobel(
        image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    # Convert the gradients to 8-bit images
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)

    # Combine horizontal and vertical gradients
    magnitude = cv2.addWeighted(
        abs_x,
        0.5,
        abs_y,
        0.5,
        0
    )

    return magnitude


def canny_edge_detection(
        image,
        lower_threshold=100,
        upper_threshold=200):
    """
    Detect edges using the Canny edge detector.
    """

    if len(image.shape) != 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    edges = cv2.Canny(
        image,
        lower_threshold,
        upper_threshold
    )

    return edges
def binary_threshold(image, threshold=127):
    """
    Convert a grayscale image into a binary image
    using a fixed threshold.
    """

    if len(image.shape) != 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    _, binary = cv2.threshold(
        image,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return binary


def adaptive_threshold(image):
    """
    Perform adaptive thresholding on a grayscale image.
    """

    if len(image.shape) != 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    adaptive = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return adaptive