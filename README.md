# Live Facial Recognition with OpenCV

[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/GrandeVx/Live-Facial-Recognition/blob/main/LICENSE.txt)  

## Introduction

The face recognition code is designed to analyze live video feed from a webcam and identify individuals by comparing their faces to a pre-defined database. The code utilizes the FaceNet512 model for face embedding and introduces a video analysis system for fine-tuning face detection. Additionally, the Nearest Neighbors algorithm is employed to speed up the face matching process.

## Features

- **Video Analysis System**: The code incorporates a video analysis system that allows for processing videos in addition to images. This feature enables fine-tuning of face detection by extracting frames from videos and analyzing each frame individually.

- **Nearest Neighbors Algorithm**: To optimize the face matching process, the code implements the Nearest Neighbors algorithm. By training a search model using user embeddings, the algorithm efficiently finds the closest match for a given face embedding, resulting in faster and more accurate face recognition.

- **.npy Embedded Mean**: After the fine-tuning process, the code calculates the mean of the generated face embeddings for each user. These mean embeddings are then saved in a .npy file. By saving the mean embeddings, subsequent face matching tasks can be expedited, as the code only needs to compare the incoming face embeddings with the mean embeddings, reducing computational overhead.

****

## Setup 🖥️

1) clone https://github.com/serengil/deepface.git library in the folder
2) Add your folder in the "database/" directory with your images or videos
3) Run the app.py and the code will do all 

## Technologies Used

The following technologies are used in the code:

- _Python_: The programming language used for developing the face recognition code.
- _deepface_: A Python library for face analysis that provides pre-trained models and utilities for face recognition tasks.
- _OpenCV_: A popular computer vision library used for image and video processing.
- _scikit-learn_: A machine learning library in Python used for implementing the Nearest Neighbors algorithm.
- _rich_: A Python library for rich text and beautiful formatting in the console.

## Development Process

The face recognition code follows a step-by-step process to perform face embedding and matching. Here is an overview of the development process:

1. **Initializing**: The code initializes the necessary folders and files for its operation.

2. **Loading FaceNet512 Model**: The FaceNet512 model, along with its weights, is loaded. This model is responsible for generating face embeddings.

3. **Generating Face Embeddings**: For each user present in the database, the code generates face embeddings. If a user's embeddings have already been generated, they are skipped. The code processes both images and videos from the user's folder, allowing for fine-tuning of face detection.

4. **Calculating User Embeddings**: The code calculates the mean of all the embeddings generated for each user and saves it in a .npy file.

5. **Loading User Embeddings**: The code loads the user embeddings from the .npy files and prepares them for face matching.

6. **Setting up Nearest Neighbor Search Model**: The Nearest Neighbors algorithm is used to create a search model for efficient face matching. The user embeddings are used to train the model, speeding up the search process.

7. **Live Analysis**: The code captures live video from the webcam and performs real-time face recognition. It extracts faces from the frames, compares them with the user embeddings using the Nearest Neighbors algorithm, and determines the closest match.
