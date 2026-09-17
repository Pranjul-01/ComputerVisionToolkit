# Computer Vision Toolkit

A Python-based Computer Vision Toolkit developed using OpenCV for performing fundamental image processing and computer vision operations.

## Project Overview

This project provides a collection of basic image processing operations that can be applied to an input image. The toolkit demonstrates commonly used techniques in computer vision and produces the processed images as output files.

The project is designed with a modular structure so that individual image processing operations can be easily understood, tested, and extended.

## Features

The toolkit currently supports the following operations:

1. Grayscale Conversion
2. Image Resizing
3. Gaussian Filtering
4. Histogram Generation
5. Histogram Equalization
6. Sobel Edge Detection
7. Canny Edge Detection
8. Binary Thresholding
9. Adaptive Thresholding

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Matplotlib

## Project Structure

```text
ComputerVisionToolkit/
│
├── data/
│   └── sample.png
│
├── output/
│   ├── grayscale.png
│   ├── resized.png
│   ├── gaussian_blur.png
│   ├── histogram.png
│   ├── equalized.png
│   ├── equalized_histogram.png
│   ├── sobel_edges.png
│   ├── canny_edges.png
│   ├── binary.png
│   └── adaptive.png
│
├── report/
│   └── screenshots/
│
├── src/
│   ├── main.py
│   ├── image_utils.py
│   └── operations.py
│
├── .gitignore
├── README.md
└── requirements.txt
