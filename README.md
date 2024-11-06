# Pharmaceutical Drug Classification using Deep Learning
This project classifies pharmaceutical drugs into 10 categories based on images using deep learning. It leverages a publicly available dataset and employs custom CNN and pre-trained models to optimize classification accuracy.

## Project Overview
This project utilizes a Kaggle dataset[1] of 10,000 images to perform a multiclass classification of pharmaceutical drugs. The model aims to help in quickly identifying various drug types from images, which can be valuable for healthcare applications.

## Features and Process
Dataset Preparation
The dataset of pharmaceutical drug images was sourced from Kaggle and consists of 10,000 images, each labeled for one of 10 drug categories.

## Data Preprocessing
Images were preprocessed using TensorFlow, including techniques such as resizing, normalization, and augmentation (rotation, flipping, scaling). These steps improve the model’s ability to generalize by providing varied representations of the data.

## Data Splitting
The dataset was split into training (70%), validation (15%), and testing (15%) sets using Scikit-learn to ensure an effective and balanced training process.

## Model Development
A custom Convolutional Neural Network (CNN) model was developed with multiple convolutional and pooling layers. Batch Normalization and Dropout layers were added to prevent overfitting and improve model robustness.

## Transfer Learning
Pre-trained models like VGG19 and InceptionResNetV2 were integrated with Transfer Learning to enhance classification accuracy. Only the top layers of these models were fine-tuned to fit the drug classification task, making use of their learned features for better performance.

## Model Optimization
The Adam optimizer and sparse categorical cross-entropy loss function optimized the model. These choices support efficient convergence during training.

## Evaluation Metrics
Key metrics, such as Precision, Recall, F1-Score, and Accuracy, were used to evaluate model performance, providing insights into how well each model variation performed on the drug classification task.

## Hyperparameter Tuning
Hyperparameters, including batch size, learning rate, and epochs, were fine-tuned using GridSearchCV to determine the optimal configuration for the model.

## Cross-Validation
Cross-validation was applied to ensure model stability across different data subsets, improving its reliability in various scenarios.

## Performance Visualization
The training history, loss curves, and other performance metrics were visualized using Matplotlib and Seaborn. This comparison between the custom CNN and pre-trained models helped assess which approach was more effective.

## Model Saving
The trained models were saved using Keras, allowing easy deployment and testing in real-world scenarios without needing retraining.

## Deployment as a Flask Web Application
The project[2] was deployed as a Flask web application on AWS, enabling real-time predictions. The deployment used AWS EC2 instances and a Putty server to manage the application, leveraging AWS services by using EC2 for model storage and seamless operation.

## Environment and Tools Used
Programming Languages & Libraries: Python, TensorFlow, Keras, Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, PIL
Development Platform: Google Colab
Deployment Platform: AWS (EC2), Flask, Putty

## Google Drive:
[Google Drive link](https://drive.google.com/drive/folders/1PxbOSDG7qlCjZjzTRy6pSH0SyRsXlATj?usp=drive_link)
In this link, you can check all the saved pre-trained models and how the dataset is sorted. 

## Reference:
1. Dataset Credit: https://www.kaggle.com/datasets/gauravduttakiit/pharmaceutical-drug-recognition

2. Patel, R. M., Vaghani, S. D., Akilan, T., & Bin , S. (2024). The Pharmaceutical Drug Classification using Deep Learning Approaches. International Journal of Advanced Natural Sciences and Engineering Researches, 8(4), 243–253. [CLICK HERE TO CHECK PAPER](https://as-proceeding.com/index.php/ijanser/article/view/1841)

# Notice
This project and all associated code, and files are the intellectual property of Ronak Patel and documentation of this project is the intellectual property of Ronak Patel and the Team. Unauthorized copying, modification, or claiming of this work as your own is prohibited.

## Disclaimer
This repository is provided for educational purposes and personal use only. It is intended to demonstrate technical concepts and does not grant permission for commercial use or reproduction in part or whole without explicit permission from the author.

## License
Unless specified otherwise, this work is licensed under the MIT License or another suitable license that aligns with your preference.
