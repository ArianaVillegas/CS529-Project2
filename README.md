# CS529-Project2

Integrants: 
 - Ariana Villegas
 - Enrique Sobrados

This repository contains the Logistic Regression and Naive Bayes models implementations for the 20-class labels document classification dataset.

To run successfully the code, follow the following steps:

## Activate conda Environment

```bash
conda env create -n proj2 -f environment.yml
conda activate proj2
```

## Run Logistic Regression


```bash
python3 main_logistic_regression.py
```

## Run Naive Bayes


```bash
python3 main_naive_bayes.py
```
The plots are saved under the following names: 

Logistic Regression: acc_logistic_regression.png and cm_logisitic_regression.png
Naive bayes: acc_naive_bayes.png and cm_naive_bayes.png
