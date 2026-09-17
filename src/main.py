from image_utils import (
    load_image,
    get_image_info,
    save_image
)

from operations import (
    convert_to_grayscale,
    resize_image,
    apply_gaussian_filter,
    save_histogram,
    equalize_histogram,
    sobel_edge_detection,
    canny_edge_detection,
    binary_threshold,
    adaptive_threshold
)


# Input image
IMAGE_PATH = "data/sample.png"

# Output directory
OUTPUT_DIR = "output"


def main():

    try:

        # ==========================================================
        # 1. LOAD ORIGINAL IMAGE
        # ==========================================================

        image = load_image(IMAGE_PATH)

        print("\n========== ORIGINAL IMAGE ==========")
        get_image_info(image)


        # ==========================================================
        # 2. GRAYSCALE CONVERSION
        # ==========================================================

        gray_image = convert_to_grayscale(image)

        print("\nGrayscale conversion successful!")
        print(f"Grayscale Shape: {gray_image.shape}")

        save_image(
            gray_image,
            f"{OUTPUT_DIR}/grayscale.png"
        )


        # ==========================================================
        # 3. IMAGE RESIZING
        # ==========================================================

        resized_image = resize_image(
            image,
            width=640,
            height=360
        )

        print("\nImage resizing successful!")
        print(f"Resized Shape: {resized_image.shape}")

        save_image(
            resized_image,
            f"{OUTPUT_DIR}/resized.png"
        )


        # ==========================================================
        # 4. GAUSSIAN FILTERING
        # ==========================================================

        gaussian_image = apply_gaussian_filter(
            image,
            kernel_size=5
        )

        print("\nGaussian filtering successful!")

        save_image(
            gaussian_image,
            f"{OUTPUT_DIR}/gaussian_blur.png"
        )


        # ==========================================================
        # 5. HISTOGRAM GENERATION
        # ==========================================================

        save_histogram(
            gray_image,
            f"{OUTPUT_DIR}/histogram.png"
        )


        # ==========================================================
        # 6. HISTOGRAM EQUALIZATION
        # ==========================================================

        equalized_image = equalize_histogram(
            gray_image
        )

        print("\nHistogram equalization successful!")

        save_image(
            equalized_image,
            f"{OUTPUT_DIR}/equalized.png"
        )


        # Save histogram of equalized image
        save_histogram(
            equalized_image,
            f"{OUTPUT_DIR}/equalized_histogram.png"
        )


        # ==========================================================
        # 7. SOBEL EDGE DETECTION
        # ==========================================================

        sobel_image = sobel_edge_detection(
            gray_image
        )

        print("\nSobel edge detection successful!")

        save_image(
            sobel_image,
            f"{OUTPUT_DIR}/sobel_edges.png"
        )


        # ==========================================================
        # 8. CANNY EDGE DETECTION
        # ==========================================================

        canny_image = canny_edge_detection(
            gray_image,
            lower_threshold=100,
            upper_threshold=200
        )

        print("\nCanny edge detection successful!")

        save_image(
            canny_image,
            f"{OUTPUT_DIR}/canny_edges.png"
        )


        # ==========================================================
        # 9. BINARY THRESHOLDING
        # ==========================================================

        binary_image = binary_threshold(
            gray_image,
            threshold=127
        )

        print("\nBinary thresholding successful!")

        save_image(
            binary_image,
            f"{OUTPUT_DIR}/binary.png"
        )


        # ==========================================================
        # 10. ADAPTIVE THRESHOLDING
        # ==========================================================

        adaptive_image = adaptive_threshold(
            gray_image
        )

        print("\nAdaptive thresholding successful!")

        save_image(
            adaptive_image,
            f"{OUTPUT_DIR}/adaptive.png"
        )


        # ==========================================================
        # FINAL STATUS
        # ==========================================================

        print("\n========== OPERATIONS COMPLETED ==========")

        print("1. Grayscale conversion")
        print("2. Image resizing")
        print("3. Gaussian filtering")
        print("4. Histogram generation")
        print("5. Histogram equalization")
        print("6. Sobel edge detection")
        print("7. Canny edge detection")
        print("8. Binary thresholding")
        print("9. Adaptive thresholding")

        print("==========================================")

        print("\nAll image processing operations completed successfully.")


    except (FileNotFoundError, ValueError, IOError) as e:

        print(f"\nError: {e}")


# ==============================================================
# PROGRAM ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()