# Life Facial Recognition 

##Introduction
The face recognition code is designed to analyze live video feed from a webcam and identify individuals by comparing their faces to a pre-defined database. The code utilizes the FaceNet512 model for face embedding and the Nearest Neighbors algorithm for face matching.

##Technologies Used
The following technologies are used in the code:

  Python: The programming language used for developing the face recognition code.
  deepface: A Python library for face analysis that provides pre-trained models and utilities for face recognition tasks.
  OpenCV: A popular computer vision library used for image and video processing.
  scikit-learn: A machine learning library in Python used for implementing the Nearest Neighbors algorithm.
  rich: A Python library for rich text and beautiful formatting in the console.

## Development Process
The face recognition code follows a step-by-step process to perform face embedding and matching. Here is an overview of the development process:

 1. Initializing: The code initializes the necessary folders and files for its operation.

 2. Loading FaceNet512 Model: The FaceNet512 model, along with its weights, is loaded. This model is responsible for generating face embeddings.

 3. Generating Face Embeddings: For each user present in the database, the code generates face embeddings. If a user's embeddings have already been generated, they are skipped. The code processes both images and videos from the user's folder.

 4. Calculating User Embeddings: The code calculates the mean of all the embeddings generated for each user and saves it in a .npy file.

 5. Loading User Embeddings: The code loads the user embeddings from the .npy files and prepares them for face matching.

 6. Setting up Nearest Neighbor Search Model: The Nearest Neighbors algorithm is used to create a search model for efficient face matching. The user embeddings are used to train the model.

 7. Live Analysis: The code captures live video from the webcam and performs real-time face recognition. It extracts faces from the frames, compares them with the user embeddings, and determines the closest match.
