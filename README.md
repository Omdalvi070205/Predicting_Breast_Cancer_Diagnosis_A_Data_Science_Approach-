# Predicting Breast Cancer Diagnosis: A Data Science Approach

### 📖 Project Overview
This project aims to develop a machine learning model to accurately predict whether a breast cancer diagnosis is malignant (M) or benign (B) based on a set of features extracted from digitized images of a fine needle aspirate (FNA) of a breast mass. The model is built using a Logistic Regression algorithm, a fundamental classifier in data science.

### 💾 Dataset
The project utilizes the Wisconsin Breast Cancer (Diagnostic) dataset. This dataset contains 569 instances and 32 attributes. The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

### Key Features:

id: Patient ID

diagnosis: The target variable (M = malignant, B = benign)

radius_mean, texture_mean, perimeter_mean, area_mean, etc.

### 🛠️ Requirements
To run this project, you need Python and the following libraries installed:

pandas

numpy

matplotlib

seaborn

scikit-learn

### You can install them using pip:

pip install pandas numpy matplotlib seaborn scikit-learn


### 🚀 How to Run
Ensure you have all the required libraries installed.

Place the Breast_cancer_dataset.csv file in the same directory as the notebook.

Open and run the Predicting Breast Cancer Diagnosis_ A Data Science Approach.ipynb notebook in a Jupyter environment.

### ⚙️ Project Workflow
#### 1. Data Loading and Initial Inspection
The dataset is loaded using pandas.

Initial checks like .head(), .info(), and .shape() are performed to understand the data's structure and size.

#### 2. Exploratory Data Analysis (EDA) & Visualization
The distribution of the target variable (diagnosis) is visualized to check for class balance.

Statistical summaries are generated using .describe().

Histograms, box plots, and count plots are used to explore the distributions of key features like radius_mean, texture_mean, and perimeter_mean and their relationship with the diagnosis.

#### 3. Data Preprocessing & Feature Engineering
The unnecessary Unnamed: 32 column is dropped.

The categorical diagnosis column is converted into a numerical format (M=1, B=0) using label encoding.

The data is split into training and testing sets to prepare for model building.

#### 4. Model Building & Training
A Logistic Regression model is chosen for this classification task.

The model is trained on the training data using the .fit() method.

#### 5. Evaluation
The trained model's performance is evaluated on the test set.

Accuracy Score is used as the primary metric to measure the model's correctness.

### A Confusion Matrix is generated to provide a more detailed view of the model's predictions, showing true positives, true negatives, false positives, and false negatives.

## 📊 Results
The Logistic Regression model achieved an accuracy of approximately 95% on the test data, indicating a high level of performance in correctly classifying tumors as malignant or benign. The Gaussian Naive Bayes model Give Accuracy 97% also demonstrated high accuracy, proving to be another effective model for this task.
