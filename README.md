# Wine Quality Prediction

This project demonstrates the process of predicting wine quality based on various physicochemical properties using a Random Forest Classifier. The dataset used for this project is the Wine Quality Data Set, which contains information on red wine variants from the Portuguese "Vinho Verde" wine. The data includes various factors like acidity, chlorides, pH, sulfates, alcohol content, and more.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Collection](#data-collection)
- [Data Analysis and Visualization](#data-analysis-and-visualization)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Building a Predictive System](#building-a-predictive-system)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/yourusername/wine-quality-prediction.git
cd wine-quality-prediction
pip install -r requirements.txt
```

## Usage

To run the project, follow these steps:

1. Load the dataset.
2. Perform data analysis and visualization.
3. Preprocess the data.
4. Train the model.
5. Evaluate the model.
6. Build a predictive system.

The following code snippet demonstrates these steps:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
wine_dataset = pd.read_csv('winequality-red.csv')

# Data analysis and visualization
sns.catplot(x='quality', data=wine_dataset, kind='count')
plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)

# Correlation heatmap
correlation = wine_dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

# Data preprocessing
X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Model training
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Model evaluation
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy:', test_data_accuracy)

# Building a predictive system
input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
if (prediction[0] == 1):
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')
```

## Data Collection

The dataset is loaded into a Pandas DataFrame from a CSV file. The dataset contains 1599 rows and 12 columns, with no missing values.

## Data Analysis and Visualization

Various visualizations are created using Seaborn and Matplotlib to understand the distribution and correlation of the features in the dataset. Key visualizations include count plots for wine quality and bar plots for volatile acidity and citric acid versus wine quality.

## Data Preprocessing

The data is split into features (X) and labels (Y). The labels are binarized to classify wine quality as good (1) if the quality is 7 or higher, and bad (0) otherwise. The data is then split into training and testing sets.

## Model Training

A Random Forest Classifier is trained on the training data. The model is then used to predict wine quality on the testing data.

## Model Evaluation

The accuracy of the model is evaluated using the accuracy score metric. In this example, the model achieves an accuracy of 0.925 on the test data.

## Building a Predictive System

A predictive system is built to classify the quality of wine based on user input. The input data is reshaped and passed through the trained model to predict whether the wine quality is good or bad.

