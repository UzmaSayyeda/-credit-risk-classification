# Credit Risk Analysis Report
## Overview  
- The objective was to train and evaluate a machine learning model to assess loan risk. The goal is to build a strong machine-learning model that can determine the creditworthiness of borrowers. The analysis aims to improve our understanding of loan risk factors, leading to a more informed and data-driven approach to identifying borrowers.
- The dataset used is from a historical lending activity from a peer-to-peer lending services company.
- The data set consisted of the following features
    - `loan_size`
    - `interest_rate`
    - `borrower_income`
    - `debt_to_income`
    - `num_of_accounts`
    - `derogatory_marks`
    - `total_debt`
    - `loan_status`
- Our target column was `loan_status`, which we were trying to predict.
- For the machine learning model we went with `LogisticRegression` which is a type of classification algorithm and later used `RandomOverSampler` which over-samples the minority class(es) by picking samples at random with replacement, this was done to balance both categories of the target variable.
## Results
To understand how our machine-learning model performed, we can look at some metrics

- Machine Learning Model 1:
- For the first model, we separated our dataset into dependent and independent variables, then split them into train and test data after which we fitted the train data to our model. Once the data was fitted we made some predictions using one of the test data and finally evaluated our model using the predictions and the remaining test data. 
  - Accuracy: The predicted accuracy for our first model is 95%
  - Precision: The precision for `Healthy Loan` or `0` is 100% and for `High-risk Loan` or `1` is 85%
  - Recall: The recall for `Healthy Loan` or `0` is 99% and for `High-risk Loan` or `1` is 91%

- Machine Learning Model 2:
- For our second model, we used `RandomOverSampler` to randomly duplicate examples in the minority class, in our case it was `0` or `High-risk Loan`. The `RandomOverSampler` is only applied to the training dataset after which we fit it to the `LogisticRegression` model, and follow the same steps to evaluate.
  - Accuracy: The predicted accuracy for our first model is 99%
  - Precision: The precision for `Healthy Loan` or `0` is 100% and for `High-risk Loan` or `1` is 84%
  - Recall: The recall for `Healthy Loan` or `0` is 99% and for `High-risk Loan` or `1` is 99%

### Summary
- Our logistic machine learning model seems to perform better when the data is resampled, this could be due to the fact that our target variable was unbalanced which led to a decrease in accuracy for the original data. 
- The accuracy for our first model was 95% while for the second it was 99%
- The second model performs better at predicting, since the recall for `High-risk Loan` is 99% while for the first model, it was 91% 
- To pick between the two possible outcomes, there is always going to be a trade-off between false positives and false negatives, for risk management, it will be better to go with high-risk loans and for risk-aversion, it is better to go with healthy loans.
- My recommendation is to evaluate other machine learning models to compare the metrics between them and this model. 
- Both models evaluated in this challenge have more than 100 false positives, this could be lowered by fine-tuning parameters in the model.  
  
  
  
### Reference
[Results](https://neptune.ai/blog/balanced-accuracy)  
[RandomOverSampler](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/)  
[Summary](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9552691/)  
[sklearn](https://scikit-learn.org/stable/index.html)
