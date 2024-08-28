# 2D Curve Detection and Analysis Toolkit

This project provides a comprehensive toolkit for detecting, analyzing, and visualizing 2D curves. It uses various Python libraries like numpy, matplotlib, scipy, and shapely to process curve data, detect regular shapes, and analyze symmetry. The toolkit includes clustering and regularization techniques to transform raw curve data into refined, symmetrical shapes.

## Project Overview

The toolkit processes curves from CSV files, extracts polylines, detects and regularizes shapes, and identifies symmetry. It is designed to handle fragmented curves and transform them into complete, regular shapes, making it ideal for applications involving curve analysis and shape recognition.

## File Descriptions

1. **`curvetopia.py`**: 
   - Main script that takes CSV files containing curve data and initiates the processing workflow.

2. **`read_and_visualise.py`**:
   - Reads CSV files, processes the data to extract polylines, and visualizes the curves using matplotlib.

3. **`fragment_detection.py`**:
   - Takes extracted polylines and clusters them using DBSCAN to form closed shapes, handling fragmented inputs effectively.

4. **`regularization.py`**:
   - Analyzes the clustered polylines to detect regular 2D shapes (e.g., circles, rectangles) using cubic splines for smooth regularization.

5. **`save_and_visualise.py`**:
   - Refines the detected shapes, corrects them based on the regularized polylines, and saves the results. Visualizes the final shapes for verification.

6. **`symmetry.py`**:
   - Detects horizontal and vertical symmetry within the detected shapes using geometric and spatial analysis techniques.

## Getting Started

To get started with the toolkit, clone the repository and install the required dependencies listed in the `requirements.txt` file.

```bash
git clone https://github.com/yourusername/2d-curve-detection.git
cd 2d_curve_detect
pip install -r requirements.txt
