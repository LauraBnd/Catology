# Catology Project: Cat Breed Classification

## Project Overview

This project aims to classify cat breeds based on a variety of factors gathered from a dataset of cat owners. The goal is to create a machine learning model that predicts a cat's breed based on attributes such as living conditions, time spent outdoors, and interaction with the owner. The dataset includes both categorical and numerical features related to each cat's environment and behavior.

## Dataset

The dataset consists of survey responses from cat owners, providing detailed information about various attributes related to their cats. The features in the dataset are:

- **Time Spent**: The amount of time the cat spends outdoors (1/2/3/4/5 - Never, Less than an hour, 1-5 hours, More than 5 hours, All the time).
- **Shy**: A rating of how shy the cat is (scale of 1-5).
- **Calm**: A rating of how calm the cat is (scale of 1-5).
- **Fearful**: A rating of how fearful the cat is (scale of 1-5).
- **Intelligent**: A rating of how intelligent the cat is (scale of 1-5).
- **Affectionate**: A rating of how affectionate the cat is (scale of 1-5).
- **Friendly**: A rating of how friendly the cat is (scale of 1-5).
- **Independent**: A rating of how independent the cat is (scale of 1-5).
- **Dominant**: A rating of how dominant the cat is (scale of 1-5).
- **Aggressive**: A rating of how aggressive the cat is (scale of 1-5).
- **Predictable**: A rating of how predictable the cat's behavior is (scale of 1-5).
- **Distracted**: A rating of how distracted the cat is (scale of 1-5).
- **Vocal**: A rating of how vocal the cat is (scale of 1-5).
- **Hair**: Indicator for whether the cat has fur (0 - No fur, 1 - With fur). This is particularly important for identifying Sphynx cats.
- **Pointy Ears**: Indicator for whether the cat has pointy ears (0 - No, 1 - Yes). This is particularly important for identifying Maine Coon cats.
- **Pattern**: The pattern of the cat's coat (e.g., striped, solid).
- **Gray coat**: Indicator for whether the cat has a gray coat (0 - No, 1 - Yes). This is particularly important for identifying Chartreux cats.
- **Limp Body**: Indicator for whether the cat has a limp body (0 - No, 1 - Yes). This is particularly important for identifying Ragdoll cats.
- **Size**: The size of the cat (e.g., small, medium, large).

### Cat Breeds

The following breeds are included in the dataset:

- **Bengal**: Abbreviation: `BEN`
- **Birman**: Abbreviation: `BIRM`
- **British Shorthair**: Abbreviation: `BRI`
- **Chartreux**: Abbreviation: `CHA`
- **European Shorthair**: Abbreviation: `EUR`
- **Maine Coon**: Abbreviation: `MCO`
- **Persian**: Abbreviation: `PER`
- **Ragdoll**: Abbreviation: `RAG`
- **Savannah**: Abbreviation: `SAV`
- **Sphynx**: Abbreviation: `SPH`
- **Siamese**: Abbreviation: `SIA`
- **Turkish Angora**: Abbreviation: `TUV`

## Description and Comparison

### Description:
This project aims to create a machine learning model that predicts a cat's breed based on attributes such as the cat's living environment, time spent outdoors, and frequency of hunting behavior. These factors are believed to influence the breed and behavior of cats, providing a basis for classification. The dataset is populated with detailed information from cat owners, making it a rich source for machine learning models.

The model will leverage categorical and numerical data from the survey responses, such as the type of housing the cat lives in and the time it spends outdoors. Additionally, the model will consider the frequency of hunting behavior (birds and small mammals), which varies between breeds and may be critical for prediction accuracy. The inclusion of an indicator for the presence of fur is especially useful for identifying breeds like the Sphynx, which is hairless.

### Comparison:
Several machine learning models will be trained and compared to determine which provides the most accurate classification of cat breeds. The comparison includes:

- **Decision Trees**: A simple and interpretable model, useful for identifying relationships between categorical variables (like the type of housing and outdoor time).
- **Random Forest**: An ensemble method that aggregates multiple decision trees, often providing better accuracy by reducing overfitting.
- **Logistic Regression**: A linear model that may perform well if there are clear linear relationships between features and the target variable.
- **Neural Networks**: A more complex model that can capture non-linear relationships, which may perform well with a larger dataset and more intricate patterns between the features and the cat's breed.

Each model will be evaluated based on **accuracy**, **precision**, **recall**, and **F1-score**. The models will be fine-tuned using cross-validation and hyperparameter optimization to achieve the best results.

## Objective

The main objective of this project is to build a machine learning model that can accurately predict the breed of a cat based on the features provided. The project involves the following key tasks:

- Data Preprocessing: Cleaning and preparing the data for training.
- Feature Engineering: Creating relevant features for classification.
- Model Development: Implementing classification models, such as decision trees, SVM, or neural networks.
- Model Evaluation: Assessing the model's performance using appropriate metrics like accuracy, precision, and recall.

## Approach

1. **Data Exploration and Cleaning**: 
   - Handle missing values, if any.
   - Convert categorical variables into numerical format (e.g., using one-hot encoding).
   - Normalize or scale numerical features to ensure uniformity.

2. **Modeling**:
   - Train and evaluate multiple models for breed classification.
   - Experiment with different algorithms (e.g., Logistic Regression, Random Forest, or Neural Networks).
   - Fine-tune the models using cross-validation and hyperparameter optimization.

3. **Evaluation**:
   - Split the dataset into training and testing sets.
   - Evaluate models using accuracy and other relevant metrics.
   - Compare model performance and select the best-performing model.

## Results

After training the model, the prediction accuracy is evaluated based on a separate test set. Additionally, a comparison of the model's predictions with the true labels will help to assess its reliability. The features, including the behavior and environmental factors of the cat, are used to provide insights into the most important factors influencing breed classification.

## Installation

To run this project, ensure that you have Python 3.x installed along with the necessary dependencies.

### Required Libraries

- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn

You can install the required libraries using `pip`:

```bash
pip install pandas scikit-learn numpy matplotlib seaborn
