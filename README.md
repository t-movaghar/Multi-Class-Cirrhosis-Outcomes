# Multi-Class-Cirrhosis-Outcomes Challenge
https://www.kaggle.com/competitions/playground-series-s3e26

## Overview

The data consists of thousands of patient metrics, including information on their sex, their cholesterol, copper levels, and more. My target variable is the patient "status," which includes C: Survived, CL: Survived w/ liver transplant, and D: Died. My goal is to use the data provided to predict whether a patient has died, survived, or survived with a liver transplant, based on the features provided.

The best performing model was the RandomForest classifier. Performance was satisfactory after data preprocessing.

### Data

* Data:
  * Type: CSV
    * Input: Two CSV files (training, testing) containing 17 clinical features for predicting survival state of patients with liver cirrhosis.
  * Size: All files total to approximately 1.39 MB
  * Target Variable: Status - "C" (Survived), "CL" (Survived w/ transplant), "D" (Died)
 
#### Preprocessing / Clean up

Non-necessary features were removed before model training (Patient ID, N_Days, Status(Target)). Data contained no null values, but most numerical variables were heavily right-tailed.
This was accounted for using a log transformation. Then, scikitlearn's Robust Scalar was applied. Categorical features were label-encoded.

#### Data Visualization

Numerical features before scaling/normalization, Separated by Stage:
![image](https://github.com/t-movaghar/Multi-Class-Cirrhosis-Outcomes/assets/123412483/80ddf9eb-dcd5-407f-8ca2-f9f1ae496eda)

Numerical features after scaling/normalization, Separated by Stage:
![image](https://github.com/t-movaghar/Multi-Class-Cirrhosis-Outcomes/assets/123412483/e276c64f-ed90-4e7a-9311-58b4a824b261)

* There seems to be a lot of features that are distinguishable between classes. 
  * Class C (Blue) tends towards lower levels of Bilrubin, Copper, SGOT.
  * Class CL (Green) Tends towards lower age.
  * Class D shows higher age and lower platelet count.
* CLASS IMBALANCE: Most are Class C or D.

Correlation matrix of all features:
![image](https://github.com/t-movaghar/Multi-Class-Cirrhosis-Outcomes/assets/123412483/92be35ac-9b28-42c0-b62a-7ef763e3b7ff)

* Copper and Bilirubin have the strongest positive correlation.

### Problem Formulation

My data consisted of 17 clinically relevant features, with a mix of both categorical and numerical variables.
I chose to use RandomForest classifier because its one of the more interpretable ML classififcation models, and it is robust to outliers, which my data still contained even after transforamtions were applied.

I also attempted to construct two simple, sequential neural network models, but their performance was outweighed by the RandomForest classifier. 
Neural networks are generally better suited to higher-dimentional, more complex datasets. RandomForest was more appropriate.

### Training and Model Performance

Data was split, using 80% for training and 20% for testing. Class weights were calculated and applied to the model in order to combat the class imbalance. N_estimators = 300.

```
Accuracy: 0.8311195445920304

Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.94      0.88       966
           1       0.64      0.17      0.27        52
           2       0.85      0.71      0.77       563

    accuracy                           0.83      1581
   macro avg       0.77      0.61      0.64      1581
weighted avg       0.83      0.83      0.82      1581

```

Confusion matrix for the Random Forest Model: 
![image](https://github.com/t-movaghar/Multi-Class-Cirrhosis-Outcomes/assets/123412483/881180d0-646f-4c52-8601-65cc535b1268)
  
One vs Rest ROC curves and AUC scores:
![image](https://github.com/t-movaghar/Multi-Class-Cirrhosis-Outcomes/assets/123412483/b222a435-6248-489e-a3ff-7f240891d819)

* ROC Curves suggest poorer performance when predicting class 1 ("CL") , which was expected, given the class imbalance.

### Model Deployment

Model was submitted to Kaggle and recieved a score of 2.17961 (Log Loss Metric).
The closer a model's score is to 0, the better performing the model.
My model was generally good at distinguishing between classes "C" and "D", but it had difficulty identifying class "CL"

### Conclusion/Future Work

The ability to predict patient outcomes of liver cirrhosis using features like blood bilirubin, copper levels, and other biomarkers is crucial for the development of early intervention methods and personalized treatment plans. 
As healthcare moves towards more data-driven and preventative approaches to medicine, predictive models can be a powerful tool in helping healthcare professionals prioritize care for those at higher risk, improve patient quality of life, and educate patients on their prognosis.

Considering the state of my model now, while its overall accuracy is high, it had difficulty identifying class "CL", which was indicated in its poor recall score. In a future attempt, I'd like to include the use of SMOTE to deal with my class imbalance.




