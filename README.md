
# Video Anomaly Detection using MRF

# Overview

This project focuses on video anomaly detection by integrating Markov Random Fields (MRFs) and autoencoders for improved accuracy and efficiency. The model leverages the spatial-temporal capabilities of MRFs to enhance anomaly differentiation in video content. By combining MRFs with autoencoders, the system generates a comprehensive video representation, improving anomaly detection performance.

The approach is tested on the AVENUE dataset, demonstrating a 60% reduction in computation time compared to conventional frame extraction methods. 
Below is an overview of the analysis, along with sample outputs and results. This project was done in Nov' 2023.

---


## Publication

- This paper was presented in the “2024 5th International Conference for Emerging Technology (INCET)”
- Link to the IEEE Publication : https://ieeexplore.ieee.org/abstract/document/10593597


---

## Block Diagram

- The below block diagram gives an overview of the overall funtionality of the implemented project
<p align="center">
  <img src="https://i.postimg.cc/LsN5Kt78/Picture2.png" alt="App Screenshot" width="400">
</p>


---

## Features

- **MRF-Based Image Segmentation**: Utilizes Markov Random Fields (MRFs) for sophisticated spatial-temporal modeling in video anomaly detection. Unlike traditional methods (HOG, LBP), MRFs capture contextual relationships between pixels and frames, enhancing anomaly detection accuracy.


- **Autoencoder-Based Feature Extraction**: Integrates autoencoders to learn compact representations of normal patterns, reconstruct video sequences, and detect anomalies. This reduces inconsistencies in extracted frames and improves the robustness of anomaly detection.

- **Sliding Window Temporal Analysis**: Implements a sliding window technique to divide segmented frames into chronological sequences. This enhances motion encoding, allowing the model to detect deviations from normal patterns effectively.
<p align="center">
  <img src="https://i.postimg.cc/d1F8VjtP/Picture3.png" alt="App Screenshot" width="300">
</p>


- **Better Computational time**: The novelty of the proposed framework also relies on that fact that this method shows a 60% lesser computation time compared to the conventional frame extraction method
<p align="center">
  <img src="https://i.postimg.cc/NFkYwYCv/Picture4.jpg" alt="App Screenshot" width="500">
</p>

---

## Tech Stack

- **Python** – Core language for implementing the video anomaly detection system.
- **OpenCV** – Used for frame extraction and image preprocessing.
- **Autoencoders (Deep Learning** – Utilized for feature extraction, reconstruction, and anomaly detection.
- **Matplotlib & Seaborn** – Used for visualizing anomaly detection results.
- **Markov Random Fields (MRFs)** - Applied for spatial-temporal image segmentation.

---


## Installation

1. **Load the AVENUE Dataset**:
- Download the dataset: https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
2. Adjust all the file paths with respect to your local file system 

3. Run the files `frames.py` and `spatio_temporal.py` to get the frames and spatio temporal features for all the videos

4. Peform the segmentation by running `mrf_segmentation.py`

5. Run `autoencoder.py` to get all the anomaly frames for the input videos


---


## Running Tests

The project can be implemented and tested to verify funtionality

