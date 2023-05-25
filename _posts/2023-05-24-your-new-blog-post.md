## This is my first blog post

Hello, this is my first post. Here I am testing some code:

```tsql
 SELECT *
 FROM sys.tables
 WHERE [name] = 'SomeTable'
```

```python
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from numpy import where

car_hacking_df = pd.read_csv('/content/car_hacking_data/clean_fuzzy_dataset.csv')

features = ['Timestamp', 'DLC', 'CAN_ID', 'Data']
X = car_hacking_df.loc[:, features].values
X_scaled = StandardScaler().fit_transform(X)

class_le = LabelEncoder()
y = class_le.fit_transform(car_hacking_df['Flag'].values)
counter = Counter(y)
print(f"Before applying NearMiss3: {counter}")
```
