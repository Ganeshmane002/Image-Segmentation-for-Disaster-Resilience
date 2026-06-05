# Image Segmentation for Disaster Resilience using U-Net

## Overview

This project develops a deep learning-based image segmentation model to identify affected regions and building footprints from satellite imagery captured during natural disasters such as floods, earthquakes, hurricanes, and wildfires.

The model assists disaster response teams by automatically detecting structures and impacted areas, enabling faster damage assessment and resource allocation.

---

## Problem Statement

Develop a deep learning-based image segmentation model to detect and classify affected regions from satellite imagery after natural disasters, aiding emergency response and disaster management.

---

## Dataset

Dataset Used: xBD (xView2 Building Damage Assessment Dataset)

The dataset contains:

- Satellite images before and after disasters
- JSON annotation files containing building polygons
- Disaster types:
  - Floods
  - Hurricanes
  - Wildfires
  - Earthquakes

Dataset Size:
- Images: 5602
- Labels: 5598 JSON annotation files

---

## Project Workflow

### Data Collection
- Downloaded xBD disaster dataset
- Extracted satellite images and annotations

### Data Preprocessing
- Converted JSON polygon annotations into segmentation masks
- Resized images to 256×256
- Normalized image pixel values
- Created binary masks

### Exploratory Data Analysis
- Image visualization
- Mask visualization
- Label distribution analysis

### Model Development
- Built a U-Net based segmentation model
- Binary segmentation for building footprint detection

### Model Evaluation
Metrics Used:
- Dice Coefficient
- IoU (Intersection over Union)
- Accuracy

---

## Technologies Used

### Programming Language
- Python

### Libraries
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Shapely

### Tools
- Google Colab
- Jupyter Notebook
- GitHub
- Streamlit

---

## Model Architecture

U-Net Architecture

Input Shape:
256 × 256 × 3

Layers:
- Conv2D
- MaxPooling2D
- Bottleneck Layer
- UpSampling2D
- Conv2D Output Layer

Output:
256 × 256 Segmentation Mask

---

## Results

### Sample Output

Input Satellite Image → Ground Truth Mask → Predicted Mask

### Performance

- IoU Score: 0.20+
- Dice Score: Improved after mask generation from JSON annotations
- Successfully segmented building regions from satellite imagery

---

## Project Structure
Image-Segmentation-Disaster-Resilience/

│
├── app.py
├── disaster_segmentation_unet.h5
├── requirements.txt
├── README.md
└── Image Segmentation for Disaster Resilience.ipynb
---

## Future Improvements

- ResNet Encoder U-Net
- Attention U-Net
- Multi-class Damage Classification
- Disaster Severity Estimation
- Real-Time Deployment

---

## Author

Ganesh Mane

