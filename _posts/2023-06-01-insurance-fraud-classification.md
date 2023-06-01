# Insurance Fraud Data Analysis and Classification

## Introduction
There are many other Kaggle projects that have investigated insurance fraud using the dataset mentioned later, but they are severely lacking. The results provided by the authors are very poor and at times are the worst when it comes to fraud classification. Moreover, the conclusions are non-existent: [9, 10]. There’s no effort towards understanding the domain knowledge of the insurance fraud dataset: they just take a model and apply it to the data—sometimes they will use ten or more models a la [10], but how can they use a model without understanding how the model works. There is a very bare minimum explanation of why the authors in the Kaggle notebooks choose to do things the way they do. 

My project hopes to address these issues: I will explore a singular Standard Vector Machine (SVM) model. I will elucidate on results, conclusions, and model parameters. The goal for this project is to have a well-written deeply explored analysis of the data with a robust discussion of metrics and a working model.

## Preprocessing and Investigation

The dataset used for the analysis is from Kaggle, Auto Insurance Claims Data (bit.ly/3JwvwTk). It is not a “perfect” dataset as there is no description of what each column means; currently, the dataset consists of 40 columns and 1,000 rows. Tools I will use are popular modules from Python such as pandas, NumPy, seaborn, scikit-learn, matplotlib, mlextend, and imbalanced-learning. The code for the project is available in a Google Collaboratory Jupyter notebook (bit.ly/408gadf); however, it is not as organized as this document. Also, some codes for plotting graphs are left out but are available in the notebook.

The document is structured in this manner: variables, columns, and functions will be italicized. In hopes of preventing confusion between variables and functions: functions include “()” parentheticals. Code will be included and if output is needed it will immediately follow. Long URLs that take up space are shortened using bit.ly. Output from the code that takes up a large amount of vertical real estate are snipped and horizontally stitched. 

Before I begin the analysis, the best thing to do is to preprocess the data for null or missing values as per machine learning purposes as well as analysis. In Fig. 1, there are null values for the columns: `collision_type`, `property_damage`, `police_report_avaliable`, `_c39`; the last column isn’t shown in the output since it has been dropped beforehand after seeing that all it offered is null values.

```python
insurance_claims_df = insurance_claims_df.drop('_c39', axis=1)
```

```python
insurance_claims_df.isna().sum()
```

<p align="center">
  <img src="https://github.com/miahj1/miahj1.github.io/assets/84815985/7f7ce06a-6d5c-407e-976e-b1667953b2f1">
</p>
Figure 1: Stitched output of all the summed null values in each column.<br><br>


The column `police_report_avaliable` has missing data where each one is inputted as “?”—prior to the imputation I have converted each “?” value to NaN using Numpy’s `replace()` function.

```python
import numpy as np

insurance_claims_df = insurance_claims_df.replace('?', np.NaN)
```

These missing values are imputed using the mode of the rows and after this step the data can now be analyzed since all the null values have been removed; it's also possible just to exclude
them from any EDA analyses but this step is required for later application of machine learning.

```python
insurance_claims_df = insurance_claims_df.fillna(insurance_claims_df.mode().iloc[0])
```

The Kaggle data also did not include the types of values that are given to each categorical column: the code below filters the data types of each column in the data frame by the parameter “object” and prints every unique value. 

```
python
cat_cols = min_df.select_dtypes(include=['object'])

for col in cat_cols:
  print(f"{col}: {cat_cols[col].unique()}")
```

In this instance, `min_df` is a modified version of the insurance dataset: the modifications will be mentioned later. Table. 1 gives an overview of each column’s unique options where each entry is separated by a space because of shear laziness on my part—certain columns are left out which will be explained later. 

Table 1: Features that have multiple unique entries for the `insurance_claims` dataset.

| Features    | Unique Entries |
| -------- | ------- |
| policy_csl  | 250/500, 100/300, 500/1000 |
| insured_sex | MALE, FEMALE|
| insured_education_level    | MD, PhD, Associate, Masters, High School, College, JD |
| insured_occupation | craft-repair, machine-op-inspct, sales, armed-forces, tech-support, prof-specialty, other-service, priv-house-serv, exec-managerial, protective-serv, transport-moving, handlers-cleaners, adm-clerical, farming-fishing |
