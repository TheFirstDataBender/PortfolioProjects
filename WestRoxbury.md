OLUYINKA OGUNDIPE (The First Data Bender)


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

```


```python
pip install dmba

```

    Requirement already satisfied: dmba in c:\users\otunba k ogundipe\anaconda3\lib\site-packages (0.1.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
from dmba import regressionSummary
```


```python
housing_df = pd.read_csv('WestRoxbury.csv')
```


```python
housing_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TOTAL VALUE</th>
      <th>TAX</th>
      <th>LOT SQFT</th>
      <th>YR BUILT</th>
      <th>GROSS AREA</th>
      <th>LIVING AREA</th>
      <th>FLOORS</th>
      <th>ROOMS</th>
      <th>BEDROOMS</th>
      <th>FULL BATH</th>
      <th>HALF BATH</th>
      <th>KITCHEN</th>
      <th>FIREPLACE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.00000</td>
      <td>5802.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>392.685715</td>
      <td>4939.485867</td>
      <td>6278.083764</td>
      <td>1936.744916</td>
      <td>2924.842123</td>
      <td>1657.065322</td>
      <td>1.683730</td>
      <td>6.994829</td>
      <td>3.230093</td>
      <td>1.296794</td>
      <td>0.613926</td>
      <td>1.01534</td>
      <td>0.739917</td>
    </tr>
    <tr>
      <th>std</th>
      <td>99.177414</td>
      <td>1247.649118</td>
      <td>2669.707974</td>
      <td>35.989910</td>
      <td>883.984726</td>
      <td>540.456726</td>
      <td>0.444884</td>
      <td>1.437657</td>
      <td>0.846607</td>
      <td>0.522040</td>
      <td>0.533839</td>
      <td>0.12291</td>
      <td>0.565108</td>
    </tr>
    <tr>
      <th>min</th>
      <td>105.000000</td>
      <td>1320.000000</td>
      <td>997.000000</td>
      <td>0.000000</td>
      <td>821.000000</td>
      <td>504.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>325.125000</td>
      <td>4089.500000</td>
      <td>4772.000000</td>
      <td>1920.000000</td>
      <td>2347.000000</td>
      <td>1308.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>375.900000</td>
      <td>4728.000000</td>
      <td>5683.000000</td>
      <td>1935.000000</td>
      <td>2700.000000</td>
      <td>1548.500000</td>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>438.775000</td>
      <td>5519.500000</td>
      <td>7022.250000</td>
      <td>1955.000000</td>
      <td>3239.000000</td>
      <td>1873.750000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1217.800000</td>
      <td>15319.000000</td>
      <td>46411.000000</td>
      <td>2011.000000</td>
      <td>8154.000000</td>
      <td>5289.000000</td>
      <td>3.000000</td>
      <td>14.000000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>2.00000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#this replace the space between the columns name with underscore '_'
housing_df.columns = [s.strip().replace(' ', '_') 
   for s in housing_df.columns]
```


```python
housing_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TOTAL_VALUE</th>
      <th>TAX</th>
      <th>LOT_SQFT</th>
      <th>YR_BUILT</th>
      <th>GROSS_AREA</th>
      <th>LIVING_AREA</th>
      <th>FLOORS</th>
      <th>ROOMS</th>
      <th>BEDROOMS</th>
      <th>FULL_BATH</th>
      <th>HALF_BATH</th>
      <th>KITCHEN</th>
      <th>FIREPLACE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.000000</td>
      <td>5802.00000</td>
      <td>5802.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>392.685715</td>
      <td>4939.485867</td>
      <td>6278.083764</td>
      <td>1936.744916</td>
      <td>2924.842123</td>
      <td>1657.065322</td>
      <td>1.683730</td>
      <td>6.994829</td>
      <td>3.230093</td>
      <td>1.296794</td>
      <td>0.613926</td>
      <td>1.01534</td>
      <td>0.739917</td>
    </tr>
    <tr>
      <th>std</th>
      <td>99.177414</td>
      <td>1247.649118</td>
      <td>2669.707974</td>
      <td>35.989910</td>
      <td>883.984726</td>
      <td>540.456726</td>
      <td>0.444884</td>
      <td>1.437657</td>
      <td>0.846607</td>
      <td>0.522040</td>
      <td>0.533839</td>
      <td>0.12291</td>
      <td>0.565108</td>
    </tr>
    <tr>
      <th>min</th>
      <td>105.000000</td>
      <td>1320.000000</td>
      <td>997.000000</td>
      <td>0.000000</td>
      <td>821.000000</td>
      <td>504.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>325.125000</td>
      <td>4089.500000</td>
      <td>4772.000000</td>
      <td>1920.000000</td>
      <td>2347.000000</td>
      <td>1308.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>375.900000</td>
      <td>4728.000000</td>
      <td>5683.000000</td>
      <td>1935.000000</td>
      <td>2700.000000</td>
      <td>1548.500000</td>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>438.775000</td>
      <td>5519.500000</td>
      <td>7022.250000</td>
      <td>1955.000000</td>
      <td>3239.000000</td>
      <td>1873.750000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1217.800000</td>
      <td>15319.000000</td>
      <td>46411.000000</td>
      <td>2011.000000</td>
      <td>8154.000000</td>
      <td>5289.000000</td>
      <td>3.000000</td>
      <td>14.000000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>2.00000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Illustrate missing data procedure to enable you work with your data
# convert some rows in variable Bedroom to NA then input missing values using the median 
missingRows=housing_df.sample(10).index
housing_df.loc[missingRows,'BEDROOMS']=np.nan
print('Number of rows with valid BEDROOMS values after setting to NAN; ',housing_df['BEDROOMS'].count())
```

    Number of rows with valid BEDROOMS values after setting to NAN;  5792
    


```python
#remove rows with missing values
reduced_df=housing_df.dropna()
print('Number of rows after removing rows with missing values: ', len(reduced_df))
```

    Number of rows after removing rows with missing values:  5792
    


```python
#replace the missing values using the median of the remaining values
medianBedrooms=housing_df['BEDROOMS'].median()
housing_df.BEDROOMS=housing_df.BEDROOMS.fillna(value=medianBedrooms)
print('Number of rows with valid BEDROOMS values after filling NA values: ', housing_df['BEDROOMS'].count())
```

    Number of rows with valid BEDROOMS values after filling NA values:  5802
    


```python
housing_df.loc[0:3]  
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TOTAL_VALUE</th>
      <th>TAX</th>
      <th>LOT_SQFT</th>
      <th>YR_BUILT</th>
      <th>GROSS_AREA</th>
      <th>LIVING_AREA</th>
      <th>FLOORS</th>
      <th>ROOMS</th>
      <th>BEDROOMS</th>
      <th>FULL_BATH</th>
      <th>HALF_BATH</th>
      <th>KITCHEN</th>
      <th>FIREPLACE</th>
      <th>REMODEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>344.2</td>
      <td>4330</td>
      <td>9965</td>
      <td>1880</td>
      <td>2436</td>
      <td>1352</td>
      <td>2.0</td>
      <td>6</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>412.6</td>
      <td>5190</td>
      <td>6590</td>
      <td>1945</td>
      <td>3108</td>
      <td>1976</td>
      <td>2.0</td>
      <td>10</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Recent</td>
    </tr>
    <tr>
      <th>2</th>
      <td>330.1</td>
      <td>4152</td>
      <td>7500</td>
      <td>1890</td>
      <td>2294</td>
      <td>1371</td>
      <td>2.0</td>
      <td>8</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>498.6</td>
      <td>6272</td>
      <td>13773</td>
      <td>1957</td>
      <td>5032</td>
      <td>2608</td>
      <td>1.0</td>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing_df['TOTAL_VALUE'].iloc[0:10]
```




    0    344.2
    1    412.6
    2    330.1
    3    498.6
    4    331.5
    5    337.4
    6    359.4
    7    320.4
    8    333.5
    9    409.4
    Name: TOTAL_VALUE, dtype: float64




```python
housing_df.iloc[:,0:1]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TOTAL_VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>344.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>412.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>330.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>498.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>331.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>5797</th>
      <td>404.8</td>
    </tr>
    <tr>
      <th>5798</th>
      <td>407.9</td>
    </tr>
    <tr>
      <th>5799</th>
      <td>406.5</td>
    </tr>
    <tr>
      <th>5800</th>
      <td>308.7</td>
    </tr>
    <tr>
      <th>5801</th>
      <td>447.6</td>
    </tr>
  </tbody>
</table>
<p>5802 rows × 1 columns</p>
</div>




```python
housing_df.sample(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TOTAL_VALUE</th>
      <th>TAX</th>
      <th>LOT_SQFT</th>
      <th>YR_BUILT</th>
      <th>GROSS_AREA</th>
      <th>LIVING_AREA</th>
      <th>FLOORS</th>
      <th>ROOMS</th>
      <th>BEDROOMS</th>
      <th>FULL_BATH</th>
      <th>HALF_BATH</th>
      <th>KITCHEN</th>
      <th>FIREPLACE</th>
      <th>REMODEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1650</th>
      <td>410.4</td>
      <td>5162</td>
      <td>8226</td>
      <td>1910</td>
      <td>2551</td>
      <td>1515</td>
      <td>2.0</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Old</td>
    </tr>
    <tr>
      <th>3378</th>
      <td>523.4</td>
      <td>6584</td>
      <td>11264</td>
      <td>1865</td>
      <td>4506</td>
      <td>2816</td>
      <td>2.0</td>
      <td>10</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2920</th>
      <td>366.2</td>
      <td>4606</td>
      <td>4205</td>
      <td>1930</td>
      <td>2396</td>
      <td>1472</td>
      <td>2.0</td>
      <td>7</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>399</th>
      <td>282.5</td>
      <td>3553</td>
      <td>6000</td>
      <td>1960</td>
      <td>2116</td>
      <td>1726</td>
      <td>1.0</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2888</th>
      <td>560.2</td>
      <td>7047</td>
      <td>15690</td>
      <td>1890</td>
      <td>5052</td>
      <td>2780</td>
      <td>2.0</td>
      <td>11</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We use this here to oversample houses with over 10 rooms
weights = [0.9 if rooms > 10 else 0.01 for rooms in housing_df.ROOMS]
housing_df.sample(5, weights=weights)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TOTAL_VALUE</th>
      <th>TAX</th>
      <th>LOT_SQFT</th>
      <th>YR_BUILT</th>
      <th>GROSS_AREA</th>
      <th>LIVING_AREA</th>
      <th>FLOORS</th>
      <th>ROOMS</th>
      <th>BEDROOMS</th>
      <th>FULL_BATH</th>
      <th>HALF_BATH</th>
      <th>KITCHEN</th>
      <th>FIREPLACE</th>
      <th>REMODEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3164</th>
      <td>873.0</td>
      <td>10982</td>
      <td>19630</td>
      <td>1910</td>
      <td>6565</td>
      <td>3374</td>
      <td>2.0</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>Recent</td>
    </tr>
    <tr>
      <th>3466</th>
      <td>428.7</td>
      <td>5393</td>
      <td>6000</td>
      <td>1990</td>
      <td>2880</td>
      <td>1711</td>
      <td>2.0</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3858</th>
      <td>673.7</td>
      <td>8475</td>
      <td>11808</td>
      <td>1918</td>
      <td>5641</td>
      <td>3120</td>
      <td>2.0</td>
      <td>12</td>
      <td>6</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3871</th>
      <td>576.1</td>
      <td>7247</td>
      <td>6501</td>
      <td>1920</td>
      <td>5197</td>
      <td>3636</td>
      <td>3.0</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>Old</td>
    </tr>
    <tr>
      <th>4051</th>
      <td>626.3</td>
      <td>7878</td>
      <td>12169</td>
      <td>1913</td>
      <td>5751</td>
      <td>2942</td>
      <td>2.0</td>
      <td>12</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Old</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing_df.dtypes

```




    TOTAL_VALUE    float64
    TAX              int64
    LOT_SQFT         int64
    YR_BUILT         int64
    GROSS_AREA       int64
    LIVING_AREA      int64
    FLOORS         float64
    ROOMS            int64
    BEDROOMS         int64
    FULL_BATH        int64
    HALF_BATH        int64
    KITCHEN          int64
    FIREPLACE        int64
    REMODEL         object
    dtype: object




```python
housing_df.loc[0:10,'REMODEL']
```




    0       None
    1     Recent
    2       None
    3       None
    4       None
    5        Old
    6       None
    7       None
    8     Recent
    9       None
    10      None
    Name: REMODEL, dtype: object




```python
housing_df.REMODEL.unique()
```




    array(['None', 'Recent', 'Old'], dtype=object)




```python
print(housing_df.REMODEL.dtype)
```

    object
    


```python
#Remodel is converted to categorical variable
housing_df.REMODEL = housing_df.REMODEL.astype('category')
```


```python
print(housing_df.REMODEL.dtype)
```

    category
    


```python
#create binary dummy & drop one variable to prevent redundancy
housing_df = pd.get_dummies(housing_df, prefix_sep='_', drop_first=True)
```


```python
housing_df.columns
```




    Index(['TOTAL_VALUE', 'TAX', 'LOT_SQFT', 'YR_BUILT', 'GROSS_AREA',
           'LIVING_AREA', 'FLOORS', 'ROOMS', 'BEDROOMS', 'FULL_BATH', 'HALF_BATH',
           'KITCHEN', 'FIREPLACE', 'REMODEL_Old', 'REMODEL_Recent'],
          dtype='object')




```python
print(housing_df.loc[:, 'REMODEL_Old':'REMODEL_Recent'].head(5))
```

       REMODEL_Old  REMODEL_Recent
    0            0               0
    1            0               1
    2            0               0
    3            0               0
    4            0               0
    


```python
excludeColumns = ('TOTAL_VALUE', 'TAX')
predictors = [s for s in housing_df.columns if s 
   not in excludeColumns]
outcome = 'TOTAL_VALUE'

```


```python
print(housing_df[predictors])
```

          LOT_SQFT  YR_BUILT  GROSS_AREA  LIVING_AREA  FLOORS  ROOMS  BEDROOMS  \
    0         9965      1880        2436         1352     2.0      6         3   
    1         6590      1945        3108         1976     2.0     10         4   
    2         7500      1890        2294         1371     2.0      8         4   
    3        13773      1957        5032         2608     1.0      9         5   
    4         5000      1910        2370         1438     2.0      7         3   
    ...        ...       ...         ...          ...     ...    ...       ...   
    5797      6762      1938        2594         1714     2.0      9         3   
    5798      9408      1950        2414         1333     2.0      6         3   
    5799      7198      1987        2480         1674     2.0      7         3   
    5800      6890      1946        2000         1000     1.0      5         2   
    5801      7406      1950        2510         1600     2.0      7         3   
    
          FULL_BATH  HALF_BATH  KITCHEN  FIREPLACE  REMODEL_Old  REMODEL_Recent  
    0             1          1        1          0            0               0  
    1             2          1        1          0            0               1  
    2             1          1        1          0            0               0  
    3             1          1        1          1            0               0  
    4             2          0        1          0            0               0  
    ...         ...        ...      ...        ...          ...             ...  
    5797          2          1        1          1            0               1  
    5798          1          1        1          1            0               0  
    5799          1          1        1          1            0               0  
    5800          1          0        1          0            0               0  
    5801          1          1        1          1            0               0  
    
    [5802 rows x 13 columns]
    


```python
#
X = housing_df[predictors]
y = housing_df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

```


```python
model = LinearRegression()
model.fit(train_X, train_y)

train_pred = model.predict(train_X)
train_results = pd.DataFrame({
    'TOTAL_VALUE': train_y, 
    'predicted': train_pred, 
    'residual': train_y - train_pred
})

```


```python
train_results.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TOTAL_VALUE</th>
      <th>predicted</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024</th>
      <td>392.0</td>
      <td>387.726258</td>
      <td>4.273742</td>
    </tr>
    <tr>
      <th>5140</th>
      <td>476.3</td>
      <td>430.785540</td>
      <td>45.514460</td>
    </tr>
    <tr>
      <th>5259</th>
      <td>367.4</td>
      <td>384.042952</td>
      <td>-16.642952</td>
    </tr>
    <tr>
      <th>421</th>
      <td>350.3</td>
      <td>369.005551</td>
      <td>-18.705551</td>
    </tr>
    <tr>
      <th>1401</th>
      <td>348.1</td>
      <td>314.725722</td>
      <td>33.374278</td>
    </tr>
  </tbody>
</table>
</div>




```python
valid_pred = model.predict(valid_X)
valid_results = pd.DataFrame({
    'TOTAL_VALUE': valid_y, 
    'predicted': valid_pred, 
    'residual': valid_y - valid_pred
})

```


```python
valid_results.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TOTAL_VALUE</th>
      <th>predicted</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1822</th>
      <td>462.0</td>
      <td>406.946377</td>
      <td>55.053623</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>370.4</td>
      <td>362.888928</td>
      <td>7.511072</td>
    </tr>
    <tr>
      <th>5126</th>
      <td>407.4</td>
      <td>390.287208</td>
      <td>17.112792</td>
    </tr>
    <tr>
      <th>808</th>
      <td>316.1</td>
      <td>382.470203</td>
      <td>-66.370203</td>
    </tr>
    <tr>
      <th>4034</th>
      <td>393.2</td>
      <td>434.334998</td>
      <td>-41.134998</td>
    </tr>
  </tbody>
</table>
</div>




```python
regressionSummary(train_results.TOTAL_VALUE, 
   train_results.predicted)

```

    
    Regression statistics
    
                          Mean Error (ME) : 0.0000
           Root Mean Squared Error (RMSE) : 43.0306
                Mean Absolute Error (MAE) : 32.6042
              Mean Percentage Error (MPE) : -1.1116
    Mean Absolute Percentage Error (MAPE) : 8.4886
    


```python
regressionSummary(valid_results.TOTAL_VALUE, valid_results.predicted)

```

    
    Regression statistics
    
                          Mean Error (ME) : -0.1463
           Root Mean Squared Error (RMSE) : 42.7292
                Mean Absolute Error (MAE) : 31.9663
              Mean Percentage Error (MPE) : -1.0884
    Mean Absolute Percentage Error (MAPE) : 8.3283
    
