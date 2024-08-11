# Adobe GenSolve
Team name: Decoders

# Shape Detection and Labeling using OpenCV

This repository contains a Python script for detecting and labeling geometric shapes in an image using OpenCV. The script identifies various shapes such as triangles, rectangles, squares, pentagons, hexagons, stars, and circles/ellipses, and labels them accordingly on the image.

## Features

- **Shape Detection:** Detects and labels different geometric shapes in an image.
- **Non-Overlapping Text Positioning:** Ensures that labels do not overlap with each other.
- **Inner Shape Detection:** Identifies shapes within other shapes and labels them appropriately.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/shape-detection.git
    cd shape-detection
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Place your image in the root directory or specify the path in the script.**

## Usage

1. **Run the script:**

    ```bash
    python detect_shapes.py
    ```

2. **View the result:**

    The script will open a window displaying the image with detected shapes and their labels.

## Example

### Input Image:
![Input Image](curve.png)

### Output Image:
An image window will pop up displaying the detected shapes labeled with their names.

## Explanation of the Code

### Key Functions:

- `angle_between(v1, v2)`: Calculates the angle between two vectors.
- `has_straight_edges(approx, tolerance=10)`: Checks if all angles in the shape are close to 90 or 180 degrees.
- `find_non_overlapping_position(positions, x, y, w, h, image_width, image_height, margin=10)`: Finds a position for text that does not overlap with existing positions.
- `detect_shapes(image_path)`: Main function to detect shapes in the provided image.

### Workflow:

- Load and preprocess the image.
- Detect contours and approximate the shape.
- Classify the shape based on its edges and angles.
- Display the image with labeled shapes.


## Acknowledgments

- OpenCV documentation and tutorials
- NumPy documentation
