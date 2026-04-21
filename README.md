# Titanic Survival Classification

This project builds a machine learning model to predict passenger survival on the Titanic. Using Python and scikit-learn, it performs exploratory data analysis, data wrangling, and trains a Logistic Regression model to classify whether a passenger survived based on their demographics and ticket information.

## Project Workflow

## 1. Exploratory Data Analysis (EDA)
Analyzes a dataset of 891 passengers.

* Visualizes survival rates overall, by sex, and by passenger class using Seaborn count plots.

* Examines the distribution of passenger ages and ticket fares using histograms.

* Identifies missing data using heatmaps and summary statistics.

## 2. Data Wrangling & Preprocessing

* Handling Missing Values: Drops the heavily incomplete Cabin column and fills missing Age values with the overall mean age.

* Categorical Encoding: Converts categorical features (Sex, Embarked, and Pclass) into dummy/indicator variables for machine learning compatibility.

* Feature Selection: Drops non-predictive or redundant columns such as Name, Ticket, and PassengerId.

## 3. Model Training

* Splits the cleaned dataset into a 70% training set and a 30% testing set.

* Initializes and trains a LogisticRegression model with max_iter=1000 to ensure proper convergence.

## 4. Evaluation & Results

* Accuracy: The model achieves an overall accuracy of approximately 79.4% on the test data.

* Classification Metrics: Generates a full classification report (Precision, Recall, F1-score) and a confusion matrix to break down true/false positives and negatives.

* Feature Importance: Extracts the model's coefficients to determine which factors (like being male or traveling in 3rd class) most heavily negatively or positively influenced the odds of survival.

## Dependencies

To run this notebook, you will need the following Python libraries installed:

`pandas`

`numpy`

`seaborn`

`matplotlib`

`scikit-learn`

## How to Run

* Ensure you have the dataset file `(titanic (1).csv)` in the same directory as the Jupyter Notebook.

* Install the required dependencies using pip or conda.

* Run the notebook cells sequentially to view the data visualizations, preprocess the dataset, and train the classification model.
