# Insurance Fraud Data Analysis and Classification

## Introduction
There are many other Kaggle projects that have investigated insurance fraud using the dataset mentioned later, but they are severely lacking. The results provided by the authors are very poor and at times are the worst when it comes to fraud classification. Moreover, the conclusions are non-existent: [9, 10]. There’s no effort towards understanding the domain knowledge of the insurance fraud dataset: they just take a model and apply it to the data—sometimes they will use ten or more models a la [10], but how can they use a model without understanding how the model works. There is a very bare minimum explanation of why the authors in the Kaggle notebooks choose to do things the way they do. 

My project hopes to address these issues: I will explore a singular Standard Vector Machine (SVM) model. I will elucidate on results, conclusions, and model parameters. The goal for this project is to have a well-written deeply explored analysis of the data with a robust discussion of metrics and a working model.

## Preprocessing and Investigation

The dataset used for the analysis is from Kaggle, Auto Insurance Claims Data (bit.ly/3JwvwTk). It is not a “perfect” dataset as there is no description of what each column means; currently, the dataset consists of 40 columns and 1,000 rows. Tools I will use are popular modules from Python such as pandas, numpy, seaborn, scikit-learn, matplotlib, mlextend, and imbalanced-learning. The code for the project is available in a Google Collaboratory Jupyter notebook (bit.ly/408gadf); however, it is not as organized as this document. Also, some codes for plotting graphs are left out but are available in the notebook.

The document is structured in this manner: variables, columns, and functions will be formatted as follows, `text`. In hopes of preventing confusion between variables and functions: functions include “()” parentheticals. Code will be included and if output is needed it will immediately follow. Long URLs that take up space are shortened using bit.ly. Output from the code that takes up a large amount of vertical real estate are snipped and horizontally stitched. 

Before I begin the analysis, the best thing to do is to preprocess the data for null or missing values as per machine learning purposes as well as analysis. In Fig. 1, there are null values for the columns: `collision_type`, `property_damage`, `police_report_avaliable`, `_c39`; the last column isn’t shown in the output since it has been dropped using pandas trusty `drop()` function beforehand after seeing that all it offered is null values.

```python
insurance_claims_df = insurance_claims_df.drop('_c39', axis=1)
```

The piped function below first checks if there are null values in the dataframe using `isna()` and then the `sum()` function
add up the total amount of null values for each column.

```python
insurance_claims_df.isna().sum()
```

```
months_as_customer               0
age                              0
policy_number                    0
policy_bind_date                 0
policy_state                     0
policy_csl                       0
policy_deductable                0
policy_annual_premium            0
umbrella_limit                   0
insured_zip                      0
insured_sex                      0
insured_education_level          0
insured_occupation               0
insured_hobbies                  0
insured_relationship             0
capital-gains                    0
capital-loss                     0
incident_date                    0
incident_type                    0
collision_type                 178
incident_severity                0
authorities_contacted            0
incident_state                   0
incident_city                    0
incident_location                0
incident_hour_of_the_day         0
number_of_vehicles_involved      0
property_damage                360
bodily_injuries                  0
witnesses                        0
police_report_available        343
total_claim_amount               0
injury_claim                     0
property_claim                   0
vehicle_claim                    0
auto_make                        0
auto_model                       0
auto_year                        0
fraud_reported                   0
```
Figure 1: Output of all the summed null values in each column.<br><br>


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

```python
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
| insured_hobbies | sleeping, reading, board-games, bungie-jumping, base-jumping, golf, camping, dancing, skydiving, movies, hiking, yachting, paintball, chess, kayaking, polo, basketball, video-games, cross-fit, exercise |
| insured_relationship | husband, other-relative, own-child, unmarried, wife, not-in-family |
| incident_type | Single Vehicle Collision, Vehicle Theft, Multi-vehicle Collision, Parked Car |
| collision_type | Side Collision, Rear Collision, Front Collision |
| incident_severity | Major Damage, Minor Damage, Total Loss, Trivial Damage |
| authorities_contacted | Police, None, Fire, Other, Ambulance |
| property_damage | YES, NO |
| police_report_avaliable | YES, NO |
| fraud_reported | Y, N |

Before starting a classification project, checking the balance of the datasets does not hurt: the labels in the dataset are binary for `fraud_reported` which contains a value of Y and N—these will be later encoded to 0 and 1 for machine learning purposes. There are 247 fraudulent cases and 753 non-fraudulent cases—a bar graph is shown in Fig. 2. A clear imbalance is visible which can be a problem depending on the type of model that is used for the machine learning section, but that is not the only obstacle: the data is meager and may not be the best for machine learning later the data is resampled to increase the minority class.

![image](https://github.com/miahj1/miahj1.github.io/assets/84815985/6e6d8837-741e-4f7c-9473-8298f858929b)

Figure 2: The amount of data per class, Fraudlent being the minority class and Non-Fradulent being the majority class. <br><br>

Let’s get some domain knowledge and then analyze the dataset to find some meaningful insights before I get ahead of myself.

## Auto-Insurance Domain Knowledge

The goal for this section is to fill gaps in my knowledge regarding the dataset. The `policy_csl` and `umbrella_limit` features are features that I did not understand: the formatting for the former are numbers separated with slashes. A bit of research has gone a long way to build an understanding of this topic—the abbreviation CSL stands for Combined Single Limits where the insurance policy limits the coverage of all components of a claim to a single dollar amount. CSL maxes out the amount of money that is paid out which covers bodily injury and property damage; interestingly, the limit splits between involved parties of the accident or claim.

If let’s say a CSL policy is 500/500, the first 500 before the slash is if an accident is caused by you the insurance will pay out up to $500,000 per person to anyone that you injured while the second 500 is the total per accident payout which is limited to a maximum of $500,000. However, additional coverage may need to be added that is required by the state or insurer.

If someone were to go over the maximum provided by their coverage, the umbrella limit would kick in. The limit in-question has no applicability in the coverage of bodily injuries or property damage. But this is not the only functionality of the umbrella limit: umbrella insurance can cover legal costs in cases of libel or slander, liabilities when traveling overseas, and expenses that are related to one self’s psychological harm and mental anguish. 

Also, the `policy_deductable` is another feature that has eluded me since I am not familiar with automobile insurance deductibles: it looks like policy deductibles are paid out of pocket on a claim which would be the basic definition. However, there is a bit more to it: a higher deductible would mean more is paid out of pocket, but the car insurance rate is lower, and a lower deductible would mean the car insurance rate is higher, but less is paid out of pocket—the inverse of the former. 

I believe that another feature can be engineered after considering the expenses covered by the insurance provider, but due to time constraints that is something I will leave for the future if I do change my mind.
