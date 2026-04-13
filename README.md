<h1 align="center"> Panorama Stitching Pipeline </h1>

<p align="center">
  <img src="https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/opencv-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" alt="Google Colab">
  <img src="https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white" alt="Google Drive">
  <img src="https://img.shields.io/badge/GCC-A42E2B?style=for-the-badge&logo=gnu&logoColor=white" alt="GCC">
  <img src="https://img.shields.io/badge/pkg--config-008FBA?style=for-the-badge&logo=pkg-config&logoColor=white" alt="pkg-config">
</p>

---

## 🧰 Tools & Technologies

| Tool / Technology | Role in Project |
|---|---|
| ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=flat-square&logo=c%2B%2B&logoColor=white) **C++** | Core language for all three stitching implementations |
| ![OpenCV](https://img.shields.io/badge/opencv-%235C3EE8.svg?style=flat-square&logo=opencv&logoColor=white) **OpenCV 4** | Computer vision library — SIFT, FLANN, RANSAC, warping, blending |
| ![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54) **Python** | Notebook scripting and Colab orchestration |
| ![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat-square&logo=jupyter&logoColor=white) **Jupyter Notebook** | Interactive pipeline execution via `PanaromaStitching.ipynb` |
| ![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=googlecolab&color=525252) **Google Colab** | Cloud execution environment with GPU/CPU runtime |
| ![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=flat-square&logo=googledrive&logoColor=white) **Google Drive** | Dataset input and panorama output storage |
| ![GCC](https://img.shields.io/badge/GCC-A42E2B?style=flat-square&logo=gnu&logoColor=white) **GCC / G++** | C++ compiler used to build stitchers inside Colab |
| ![pkg-config](https://img.shields.io/badge/pkg--config-008FBA?style=flat-square) **pkg-config** | Resolves OpenCV linker flags during compilation |

---


## 📌 Overview

This repository explores and implements various approaches to **Image Stitching** and **Panorama Generation** using C++ and OpenCV. The goal is to geometrically align and seamlessly blend a sequence of overlapping photographs into a single extreme-wide-angle canvas.

The project explores stitching algorithms across **three different architectural approaches**, scaling from a completely custom pairwise homography engine, up to utilizing advanced multi-band spherical blending pipelines.

---

## 🚀 The Three Approaches

### 1. Basic Stitcher (Primary Method)
> **Source:** [`basic_stitcher.cpp`](basic_stitcher.cpp)

This is the main focal point of the repository. It implements a completely custom, hierarchical "tournament-style" stitching engine from scratch. Instead of relying on pre-built pipeline black-boxes, this method explicitly defines the math and logic for aligning images.

* **Feature Extraction:** Identifies 500 dominant keypoints per image using the SIFT (Scale-Invariant Feature Transform) algorithm.
* **Feature Matching:** Uses the FLANN (Fast Library for Approximate Nearest Neighbors) matcher to quickly cross-reference descriptors between images.
* **Connectivity Matrix:** Calculates an N x N Inlier Matrix by validating RANSAC homographies to determine which images overlap the most.
* **Hierarchical Warping:** Greedily pairs images with the highest overlap, warps one perspective space onto the other (`cv::warpPerspective`), masks the boundaries, and repeats until a single canvas remains.

### 2. Advanced Stitcher (Low-Level Detail API)
> **Source:** [`advanced_stitcher.cpp`](advanced_stitcher.cpp)

This approach leverages OpenCV's lower-level `cv::detail` stitching components to construct a professional-grade panoramic pipeline piece by piece.
* **Camera Estimation:** Utilizes Bundle Adjustment to solve camera intrinsic and extrinsic focal/rotation matrices.
* **Spherical Warping:** Casts images onto a spherical projection to prevent extreme edge stretching found in planar homography.
* **Seam Finding:** Employs Voronoi seam-finding logic to slice image overlaps exactly where visual differences are minimized.
* **Multi-Band Blending:** Uses Laplacian pyramids to smoothly merge image extremities without creating blur or ghosting artifacts.

### 3. OpenCV Native Stitcher
> **Source:** [`opencv_stitcher.cpp`](opencv_stitcher.cpp)

A reference implementation using OpenCV's built-in, abstract `cv::Stitcher::create(Stitcher::PANORAMA)` class. It acts as a benchmark to compare our custom code against fully optimized, production-ready library standards.

---

## 📸 Results Gallery

Below are the rendered generated panoramas mapped from the individual dataset frames.

### Basic Stitcher Output (Custom Hierarchical Pipeline)
*Notice the planar homography perspective effects.*

**Dataset 1:**
<p align="center"> <img src="Output/panorama_basic_1.jpg" width="800"> </p>

**Dataset 2:**
<p align="center"> <img src="Output/panorama_basic_2.jpg" width="800"> </p>

**Dataset 3:**
<p align="center"> <img src="Output/panorama_basic_3.jpg" width="800"> </p>

---

### Advanced Stitcher Output (Spherical Warping & Multi-Band Blending)
*Notice the cleaner seams and the curved spherical warping to retain proportions.*

**Dataset 1:**
<p align="center"> <img src="Output/panorama_advanced_1.jpg" width="800"> </p>

**Dataset 2:**
<p align="center"> <img src="Output/panorama_advanced_2.jpg" width="800"> </p>

**Dataset 3:**
<p align="center"> <img src="Output/panorama_advanced_3.jpg" width="800"> </p>

---

### OpenCV Core Stitcher Benchmark

**Dataset 1:**
<p align="center"> <img src="Output/panorama_opencv_stitcher_1.jpg" width="800"> </p>

**Dataset 2:**
<p align="center"> <img src="Output/panorama_opencv_stitcher_2.jpg" width="800"> </p>

**Dataset 3:**
<p align="center"> <img src="Output/panorama_opencv_stitcher_3.jpg" width="800"> </p>

---

## 🛠 Compilation and Usage in Google Colab

This project is configured to run efficiently inside Google Colab, leveraging Google Drive for data storage. You can find the full executable pipeline in the provided `PanaromaStitching.ipynb` notebook.

### 1. Setup Environment
First, mount your Google Drive to access the dataset and store the output panoramas. Then, update the package list and install the OpenCV development libraries.
```python
# In a Colab cell
from google.colab import drive
drive.mount('/content/drive')

!sudo apt-get update
!sudo apt-get install libopencv-dev
```

### 2. Prepare Output Directory
Create the destination folder in your Drive to hold the generated panoramic images:
```bash
!mkdir -p "/content/drive/MyDrive/MachineVision/Output"
```

### 3. Compilation
You can compile any of the stitchers directly inside Colab using `g++` and `pkg-config`:

```bash
# Compile the Advanced Stitcher as an example
!g++ advanced_stitcher.cpp -o app `pkg-config --cflags --libs opencv4`

# Alternatively, compile the Basic Stitcher or OpenCV Stitcher
# !g++ basic_stitcher.cpp -o app `pkg-config --cflags --libs opencv4`
# !g++ opencv_stitcher.cpp -o app `pkg-config --cflags --libs opencv4`
```

### 4. Running the App
Execute the compiled binary (`app`), passing the path to the input image folder on your Drive and specifying the output filename.

```bash
# Example: Stitching the 'office2' dataset
!./app "/content/drive/MyDrive/MachineVision/Data/office2" "/content/drive/MyDrive/MachineVision/Output/panorama_advanced_1.jpg"
```
