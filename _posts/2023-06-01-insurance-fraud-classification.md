# Insurance Fraud Data Analysis and Classification

## Introduction
There are many other Kaggle projects that have investigated insurance fraud using the dataset mentioned later, but they are severely lacking. The results provided by the authors are very poor and at times are the worst when it comes to fraud classification. Moreover, the conclusions are non-existent: [9, 10]. There’s no effort towards understanding the domain knowledge of the insurance fraud dataset: they just take a model and apply it to the data—sometimes they will use ten or more models a la [10], but how can they use a model without understanding how the model works. They also use visualization techniques that are frowned upon without a rigours understanding of why each decision is made in the graph or what they want to portray in that graph: this is a common ailment in most kaggle notebooks I've seen. There is a very bare minimum explanation of why the authors in the Kaggle notebooks choose to do things the way they do. 

My project hopes to address these issues: I will explore a singular Standard Vector Machine (SVM) model. I will elucidate on results, conclusions, and model parameters. I wil use visualizations that are readable, clean, and clearly state their reason for existing. The goal for this project is to have a well-written deeply explored analysis of the data with a robust discussion of metrics and a working model.

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
<p align="center"><strong>Figure 1:</strong> <i>Output of all the summed null values in each column.</i></p><br>


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

<p align="center">
  <img src="https://github.com/miahj1/miahj1.github.io/assets/84815985/9db6e418-777b-4608-b60d-ed5803cec456" alt="Bar graph distribution of classes.">
</p>

<p align="center"><strong>Figure 2:</strong> <i>The amount of data per class, fraudulent being the minority class and non-fraudulent being the majority class.</i></p><br>

Let’s get some domain knowledge and then analyze the dataset to find some meaningful insights before I get ahead of myself.

## Auto-Insurance Domain Knowledge

The goal for this section is to fill gaps in my knowledge regarding the dataset. The `policy_csl` and `umbrella_limit` features are features that I did not understand: the formatting for the former are numbers separated with slashes. A bit of research has gone a long way to build an understanding of this topic—the abbreviation CSL stands for Combined Single Limits where the insurance policy limits the coverage of all components of a claim to a single dollar amount. CSL maxes out the amount of money that is paid out which covers bodily injury and property damage; interestingly, the limit splits between involved parties of the accident or claim.

If let’s say a CSL policy is 500/500, the first 500 before the slash is if an accident is caused by you the insurance will pay out up to $500,000 per person to anyone that you injured while the second 500 is the total per accident payout which is limited to a maximum of $500,000. However, additional coverage may need to be added that is required by the state or insurer.

If someone were to go over the maximum provided by their coverage, the umbrella limit would kick in. The limit in-question has no applicability in the coverage of bodily injuries or property damage. But this is not the only functionality of the umbrella limit: umbrella insurance can cover legal costs in cases of libel or slander, liabilities when traveling overseas, and expenses that are related to one self’s psychological harm and mental anguish. 

Also, the `policy_deductable` is another feature that has eluded me since I am not familiar with automobile insurance deductibles: it looks like policy deductibles are paid out of pocket on a claim which would be the basic definition. However, there is a bit more to it: a higher deductible would mean more is paid out of pocket, but the car insurance rate is lower, and a lower deductible would mean the car insurance rate is higher, but less is paid out of pocket—the inverse of the former. 

I believe that another feature can be engineered after considering the expenses covered by the insurance provider, but due to time constraints that is something I will leave for the future if I do change my mind.

## Visualizing the Dataset in Python

The gender binary split brings into question: How many of the customers that are male or female commit insurance fraud and how many do not? Customers that are female who commit insurance fraud are 126 and their male counterparts are 121: this clearly shows that females commit insurance fraud at a slightly higher amount in this specific dataset. However, if we look at it the other way in-terms of the ones who do not commit insurance fraud, we find that most females about 411 of them do not while 342 males trail behind. A bar graph of the results are shown in Fig. 3.

We use boolean operators such as `==` and `&` to extract the needed information. The equality operation for fraudulent females checks if the `fraud_reported` column value is true i.e., `Y`
and the expression after the `&` operation checks if the gender of the insured in the column `insured_sex` is `FEMALE`. The `sum()` function is then used to add every occurence that meets
the criteria for our boolean expression.

```python
print(f"Fraudulent Females: {((insurance_claims_df['fraud_reported'] == 'Y') & (insurance_claims_df['insured_sex'] == 'FEMALE')).sum()} \n"
      f"Non-fraudulent Females: {((insurance_claims_df['fraud_reported'] == 'N') & (insurance_claims_df['insured_sex'] == 'FEMALE')).sum()}\n"
      f"Fraudulent Males: {((insurance_claims_df['fraud_reported'] == 'Y') & (insurance_claims_df['insured_sex'] == 'MALE')).sum()}\n"
      f"Non-fraudulent Males: {((insurance_claims_df['fraud_reported'] == 'N') & (insurance_claims_df['insured_sex'] == 'MALE')).sum()}")
```

Let's now graph the results, the `x_axis` variable is assigned the two categories for fraud classification. `y_axis_1` is assigned the values for the female customers while
`y_axis_2` is assigned the values for the male customers. The code `fig, (ax1, ax2) = plt.subplots(1, 2)` is used to establish two graphs. The function `suptitle()` is 
normally used for titling the shared plots; however, I want to title the shared x-axis so I set the `y` argument value to `-0` effectively flipping it over to the other end.
The other styling changes to the graphs are self-explanatory and won't be covered.

```python
import matplotlib.pyplot as plt

x_axis = ['Fraudulent', 'Non-Fraudulent']
y_axis_1 = [126, 121]
y_axis_2 = [411, 342]

fig, (ax1, ax2) = plt.subplots(1, 2)

# Negative value is used to flip the title to the x-axis.
plt.suptitle("Type of Fraud", y = -0, fontsize = 13.0)

ax1.set_frame_on(False)
ax1.grid(axis = 'y', alpha = 0.5)
ax1.set(ylim=(0, 450))
ax1.set_ylabel('Total Occurences of Fraud', fontsize = 13.0) 
ax1.title.set_text('Female') 
ax1.bar(x_axis, y_axis_1)

ax2.set_frame_on(False)
ax2.set_yticklabels([])
ax2.grid(axis = 'y', alpha = 0.5)
ax2.set(ylim=(0, 450))
ax2.title.set_text('Male') 
ax2.bar(x_axis, y_axis_2)
```

<p align="center">
  <img src="https://github.com/miahj1/miahj1.github.io/assets/84815985/3b31b431-3da7-4e4a-8265-b5ebd9caf36a" alt="Bar graph subplot distribution of classes for male and female customers.">
</p>

<p align="center"><strong>Figure 3:</strong> <i>Distribution of classes based on gender.</i></p><br>


Moreover, the education level feature allows me to show at what level of education customers are most likely or least likely to commit insurance fraud.
```python
education_levels = insurance_claims_df['insured_education_level'].unique()

for education in education_levels:
  print(f"{education}: {((insurance_claims_df['insured_education_level'] == education) & 
                        (insurance_claims_df['fraud_reported'] == 'Y')).sum()}")
```

Using Pandas `unique()` function, I assign the list it returns for the column `insured_education_level` to the declared variable `education_levels`: the function returns all the unique values it finds in that specific column. Next, I iterate through the `education_levels` printing the results of a boolean expression. The expression in-question looks to each education level and its connection to fraudulent activity being ‘Y’ which is added.

```python
import seaborn as sns
import matplotlib

education_levels = {'MD' : 38,
                    'Phd' : 33,
                    'Associate' : 34,
                    'Masters' : 32,
                    'High School' : 36,
                    'College' : 32,
                    'JD' : 42}

sorted_ed_levels = dict(sorted(education_levels.items(), key=lambda x: x[1]))
ax = sns.stripplot(x = sorted_ed_levels.values(), y = sorted_ed_levels.keys(), size = 8)
plt.grid(axis = 'y', alpha = 0.5)
plt.grid(axis = 'x', alpha = 0.5)

ax.set_frame_on(False)
ax.set_xlabel("Total Amount of Fraudulent Cases", fontsize = 13.0)
matplotlib.rc('ytick', labelsize=11) 
```
<p align="center">
  <img src="https://github.com/miahj1/miahj1.github.io/assets/84815985/94242eef-86ba-444d-81fd-e69c8bba91fc" alt="Cleveland dot plot of education type vs amount of fraud.">
</p>

<p align="center"><strong>Figure 4:</strong> <i>Insurance fraud committed based on insured’s level of education.</i></p><br>

The cleveland dot plot as shown in Fig. 4 reveals that customers with the education level of Juris Doctor (JD) commit the most amounts of insurance fraud while customers with masters and college degrees are on the lower end: this data does not however give an idea of why JD is the highest. 

Let's look at the age of fraudsters from both genders.

<p align="center">
  <img src="https://github.com/miahj1/miahj1.github.io/assets/84815985/ffdd3f7d-b75b-48e1-b626-1af1209c7f67" alt="Age pyramid of age vs amount of fraud including gender categorization.">
</p>

<p align="center"><strong>Figure 5:</strong> <i>Age range of customers that commit insurance fraud.</i></p><br>

Women in their early forties--just like their male counterparts--commit a huge amount of insurance fraud. There's two key differences between the groups where women in their early thirties have a second peak while men peak largely in their early forties; moreover, men in their late twenties peak earlier than later on in their lives compared to women. However, both groups mellow out from these extremes in their senior years.

Now, let's look at the code for making the Fig. 5 graph.

```python 
def dataframe_gen(gender: str, ages = insurance_claims_df['age'].unique()):
  fraud_per_age = []

  for age in ages:
    fraud_per_age.append(((insurance_claims_df['age'] == age) & 
                          (insurance_claims_df['fraud_reported'] == 'Y') &
                          (insurance_claims_df['insured_sex'] == gender)).sum())

  df = pd.DataFrame(
    dict(
        age = ages,
        fraud_freq = fraud_per_age,
        gender = gender
    )
  )

  return df

df1 = dataframe_gen("MALE")
df2 = dataframe_gen("FEMALE")
```

Before plotting the data, I define a function named `dataframe_gen()` which takes arguments `gender` and `ages`: the gender parameter is limited to two strings based on the data i.e. `MALE` or `FEMALE`. The `ages` argument is preloaded with code that returns all the unique ages in the `ages` column for our dataframe. The making of the pyramid graph requires having two dataframes. When the function is called with the appropriate parameter, an empty list named `fraud_per_age` is declared. A for loop is used to iterate through the `ages` list. The `fraud_per_age` list is then used to append or store values that meet the boolean expression which can be simplified as the age in the dataframe being equal to each age in our unique list of ages in `ages` and if those in the list of ages have commited insurance fraud and they are the specified `gender` we sent in earlier as an argument to the function. If all these conditions are true, append the total fraudlent cases to `fraud_per_age`. Purpose of the function is to generate a dataframe, a dataframe object is constructed using a dictionary with the names of the columns and their coressponding data. The function gracefully returns a dataframe. `df1` has all the values that correspond to `MALE` customers while `df2` has all the values that correspond to  the `FEMALE` customers. 

With all this setup, we can now graph the plot.

```python
import matplotlib.pyplot as plt

plt.barh(width = df1["fraud_freq"], y = df1["age"], label = "Male")
plt.barh(width = df2["fraud_freq"], y = df2["age"], left = -df2["fraud_freq"], label = "Female")
```

One of the simpliest ways to create an age pyramid is to use two horizontal bar graphs and have one face the opposite end. The first line of code creates a horizontal bar graph on the right side of the pyramid for male customers while the second line of code created a horizontal bar graph on the left side of the pyramid. This is achieved using the argument `left` which is assigned the negative argument `-df2["fraud_freq"]` causing the number line to start from negative values which isn't ideal but will be fixed later on.

```python
plt.xlim(-df2["fraud_freq"].max(), df2["fraud_freq"].max())
plt.ylabel("Age (years)", fontsize = 13)
plt.xlabel("Total Amount of Fraudulent Cases", fontsize = 13)
plt.xticks(np.arange(-12, 12, 6), labels = [12, 6, 0, 6])
plt.text(-5.5, 62, "Female", fontsize = 13)
plt.text(6.5, 62, "Male", fontsize = 13)

ax = plt.gca()
ax.set_frame_on(False)
ax.set_axisbelow(True)
ax.grid(color='gray', alpha=0.3)
```

The function `xlim` allows limiting the x-values: we take the maximum value from `df2` which is the largest max value of fraud totals. To resolve the issue with negative values appearing in the x-axis, `xticks` becomes very helpful. The `xticks` function's first argument is given `np.arange(-12, 12, 6)` which generates the number line for us e.g. `-12 -6, 0, 6, 12` and the second argument `labels` allow us to hide those pesky negative values by giving it positive values for the number line we want to see.
