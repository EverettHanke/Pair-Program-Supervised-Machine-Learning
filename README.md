# Supervised Machine Learning

## Project Overview
This project focuses on building a supervised machine learning model to predict the likelihood of a heart attack based on various health and lifestyle factors. The dataset used for this project includes features such as age, BMI, smoking status, and sleep hours. The primary goal is to preprocess the data, train a logistic regression model, and evaluate its performance using classification metrics.

## Features Used
The following features were selected for the model:
- **Sex**: Biological sex of the individual.
- **AgeCategory**: Age group of the individual.
- **BMI**: Body Mass Index, a measure of body fat based on height and weight.
- **SmokerStatus**: Whether the individual is a smoker.
- **SleepHours**: Average number of hours of sleep per night.

## Steps Performed
1. **Data Preprocessing**:
   - Removed rows with missing target values.
   - Selected relevant features for modeling.
   - Handled missing feature values by dropping incomplete rows.
   - One-hot encoded categorical variables.
   - Scaled numeric features using Min-Max Scaling.

2. **Modeling**:
   - Split the data into training and testing sets (80/20 split).
   - Trained a **Logistic Regression** model on the training data.
   - Predicted outcomes on the test set.

3. **Evaluation**:
   - Evaluated the model using the following metrics:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1 Score**
   - Visualized the confusion matrix using a heatmap.

## Results
- **Accuracy**: The model achieved an accuracy of `X.XXX` (replace with actual value).
- **Precision**: `X.XXX`
- **Recall**: `X.XXX`
- **F1 Score**: `X.XXX`

## Insights
- The most important features in the model were:
  - **AgeCategory**: Older individuals are at higher risk of heart attacks.
  - **BMI**: Obesity is a significant risk factor.
  - **SmokerStatus**: Smoking increases the likelihood of heart disease.
  - **SleepHours**: Poor sleep is associated with increased health risks.

## Recommendations
- This model should be used as a supplementary tool for identifying high-risk individuals, not as a standalone diagnostic tool.
- Regularly update the model with new data to maintain accuracy.
- Consider adding more features (e.g., cholesterol levels, blood pressure) for better predictions.

## Limitations
- The dataset may not be representative of the entire population, leading to potential biases.
- Logistic Regression assumes linear relationships, which may not fully capture complex patterns in the data.
- The model's predictions are only as good as the quality and diversity of the training data.

## How to Run the Project
1. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt