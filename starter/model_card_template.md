# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model uses a Random Forest classifier with the default parameters from Sklearn library.

## Intended Use
To predict if the income of a person is greater than $50K every year or not.

## Training Data
The training dataset: [https://archive.ics.uci.edu/ml/datasets/census+income](Census Income Data Set)

## Evaluation Data
After cleaning the data, also know as after EDA the shape of the data is (30162, 15). The data was splitted into 80% training and 20% testing data. The model was trained on the training data and evaluated on the testing data. 

## Metrics
The model was evaluated using precision, recall and f1 score.

## Ethical Considerations
This dataset is not a reliable reflection of the distribution of salaries and should not be used to make assumptions about the salary levels of particular groups of people.

## Caveats and Recommendations
To further enhance performance, one can explore hyperparameter optimization. In addition man can use the K-fold cross validation `KFold` instaed of `train_test_spilt` from sklearn library. Another valuable recommendation is to engage in feature engineering. 