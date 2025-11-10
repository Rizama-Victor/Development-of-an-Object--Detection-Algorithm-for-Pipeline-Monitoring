# Development of an Object Detection Algorithm for Crude Oil Pipeline Leakage Detection 
This repository contains the implementation of the project titled _"Development of an Object Detection Algorithm for Crude Oil Pipeline Leakage Detection"_, at the Advanced Engineering and Innovation Research Group (AEIRG), Department of Mechatronics Engineering, Federal University of Technology Minna (FUTMinna), Nigeria.

---

## 🧠 Project Overview
Pipeline systems are vital for transporting oil and gas but remain vulnerable to leaks and failures that threaten the environment. 
Traditional monitoring techniques often lack real-time detection and predictive capabilities, leading to delayed responses and expensive damage. 
To address these challenges, this project leverages computer vision and deep learning to enhance pipeline surveillance. 
It focuses on developing an intelligent object detection model capable of extracting landmark features from images and videos for real-time pipeline monitoring.

---

## 🔍 Research Objectives
- To curate a high-quality dataset suitable for training the intelligent object detection model.
- To develop an object detection model suitable for crude oil pipeline detection.
- To evaluate and assess the performance of the developed model in accurately detecting pipeline leaks under real-world conditions.

---

## ⚙️ Tools and Technologies Used

| Tool/Libraries | Purpose in the Project |
|--------------------|-----------------------|
| Python | Used as the main programming language for developing and training the computer vision model, and the algorithm. |
| OpenCV | Used for handling images, image processing, and video frame extraction. |
| YOLOv5 | Implemented for the real-time pipeline leakage detection and classification. |
| Ultralytics | Provided the implementation of YOLO used for both training and inferencing with the YOLO-based model for the pipeline leakage detection and classification. |
| Pytorch | Used for model training and performance evaluation. |
| Google Colab | Provided the virtual environment and computational resources such as GPU support for running and training the model. |
| Roboflow | Used for hosting the dataset and performing data pre-processing and preparation. |
| Display | Used for displaying inferenced images, training results and test images in the program notebook. |
| Image | Used for creating python objects representing an image. |

---

## 🏗️ Model Building and Development

The detection of pipeline leaks was achieved using a deep learning approach that involved transfer learning. The process began with capturing or obtaining images of pipelines with and without using cameras, followed by preprocessing steps like data augmentation and cleaning. During training, a pre-trained neural network specifically YOLOv5s was fine-tuned on a custom dataset of pipeline images to automatically learn and identify visual patterns that indicated leaks, such as changes in texture, color, or shape. Next, post-processing techniques such as confidence thresholding and non-maximum suppression were then applied to improve detection accuracy and reliability.

---

YOLOv5 was selected as the core deep learning architecture for this project because:

- YOLOv5 treats object detection as a single regression task, allowing it to predict bounding boxes and class probabilities in one forward pass.
- It eliminates the multiple processing stages required in traditional models like R-CNN, Fast R-CNN, and Faster R-CNN.
- It balances speed and accuracy making it capable of detecting leaks of varying sizes and in complex environments.

---

### Data Acquisition

The dataset used for this project initially consisted of 1,908 images of both leaking and non-leaking pipes collected from sources like Roboflow and Google Datasets. However, additional images were later gathered from platforms such as Kaggle and through web scraping, expanding the dataset to 10,000 images. Including non-leaking (negative) pipeline images was essential in helping the model accurately distinguish between normal and leaking pipelines in a bid to improve its overall accuracy and reliability.
