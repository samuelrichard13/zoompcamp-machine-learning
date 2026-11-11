# Mid term project : Sleep score prediction

## Problem description

Sleep plays a vital role in maintaining good health, as it allows both the body and mind to regenerate deeply. During the night, the body consolidates memory, regulates emotions, and repairs tissues, which supports better concentration and greater resistance to stress. Adequate sleep also strengthens the immune system, helping the body fight infections and prevent various illnesses. Conversely, chronic sleep deprivation can lead to persistent fatigue, mood disturbances, reduced cognitive performance, and an increased risk of cardiovascular or metabolic problems. Sleep also has a decisive impact on athletic performance: it enhances coordination, reaction speed, endurance, and muscle recovery, enabling athletes to train more effectively and reduce the risk of injury. Taking care of one’s sleep, by adopting regular habits and a restful environment, is therefore a fundamental pillar for maintaining physical, mental, and athletic health.

This project aims to predict sleep scores based on data such as duration, bedtime, and screen time before sleeping. Beyond simply predicting the score, the objective is to determine the parameters that most influence sleep quality.

## Data preparation and EDA

Here is a preview of the dataset :

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>age</th>
      <th>gender</th>
      <th>chronotype</th>
      <th>occupation</th>
      <th>bedtime_hour</th>
      <th>screen_time_before_bed_min</th>
      <th>primary_app_category</th>
      <th>screen_brightness_pct</th>
      <th>blue_light_filter_on</th>
      <th>...</th>
      <th>room_temperature_f</th>
      <th>ambient_noise_db</th>
      <th>sleep_onset_latency_min</th>
      <th>total_sleep_hours</th>
      <th>rem_sleep_pct</th>
      <th>deep_sleep_pct</th>
      <th>awakenings_count</th>
      <th>sleep_efficiency_pct</th>
      <th>next_day_fatigue_score</th>
      <th>sleep_debt_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>32</td>
      <td>Female</td>
      <td>Intermediate</td>
      <td>Student</td>
      <td>23.94</td>
      <td>102</td>
      <td>Social Media</td>
      <td>100</td>
      <td>False</td>
      <td>...</td>
      <td>73.5</td>
      <td>32.6</td>
      <td>60.3</td>
      <td>5.67</td>
      <td>15.2</td>
      <td>13.5</td>
      <td>11</td>
      <td>63.7</td>
      <td>8.85</td>
      <td>Moderate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>26</td>
      <td>Female</td>
      <td>Intermediate</td>
      <td>Student</td>
      <td>0.06</td>
      <td>14</td>
      <td>Short-Form Video</td>
      <td>87</td>
      <td>True</td>
      <td>...</td>
      <td>66.8</td>
      <td>44.3</td>
      <td>24.5</td>
      <td>7.42</td>
      <td>16.9</td>
      <td>22.3</td>
      <td>1</td>
      <td>92.9</td>
      <td>2.98</td>
      <td>No Debt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>34</td>
      <td>Male</td>
      <td>Night Owl</td>
      <td>Professional</td>
      <td>1.48</td>
      <td>58</td>
      <td>Messaging</td>
      <td>76</td>
      <td>True</td>
      <td>...</td>
      <td>68.0</td>
      <td>45.6</td>
      <td>21.9</td>
      <td>6.98</td>
      <td>20.2</td>
      <td>18.5</td>
      <td>2</td>
      <td>84.6</td>
      <td>3.90</td>
      <td>Mild</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>43</td>
      <td>Female</td>
      <td>Intermediate</td>
      <td>Professional</td>
      <td>22.42</td>
      <td>48</td>
      <td>Social Media</td>
      <td>56</td>
      <td>False</td>
      <td>...</td>
      <td>72.3</td>
      <td>42.3</td>
      <td>34.5</td>
      <td>7.67</td>
      <td>14.4</td>
      <td>17.3</td>
      <td>6</td>
      <td>79.9</td>
      <td>4.81</td>
      <td>No Debt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>25</td>
      <td>Male</td>
      <td>Early Bird</td>
      <td>Professional</td>
      <td>22.43</td>
      <td>29</td>
      <td>Streaming</td>
      <td>96</td>
      <td>True</td>
      <td>...</td>
      <td>80.1</td>
      <td>10.0</td>
      <td>37.4</td>
      <td>6.31</td>
      <td>16.6</td>
      <td>16.1</td>
      <td>0</td>
      <td>89.3</td>
      <td>3.88</td>
      <td>Mild</td>
    </tr>
  </tbody>
</table>

We convert the only boolean feature as integer with :

```python
df["blue_light_filter_on"] = df["blue_light_filter_on"].astype(int)
```
and we extract categorical and numerical variables with :

```python
categorical_variable = df.columns[df.dtypes == 'str'].drop('sleep_debt_category').tolist()
numerical_variable = df.columns[df.dtypes != 'str'].drop('next_day_fatigue_score').tolist()
```

Then, we check that ther is no empty cell with :

```python
df.isnull().sum()
```

```
user_id                       0
age                           0
gender                        0
chronotype                    0
occupation                    0
bedtime_hour                  0
screen_time_before_bed_min    0
primary_app_category          0
screen_brightness_pct         0
blue_light_filter_on          0
caffeine_intake_mg            0
caffeine_last_hour            0
physical_activity_min         0
room_temperature_f            0
ambient_noise_db              0
sleep_onset_latency_min       0
total_sleep_hours             0
rem_sleep_pct                 0
deep_sleep_pct                0
awakenings_count              0
sleep_efficiency_pct          0
next_day_fatigue_score        0
sleep_debt_category           0
dtype: int64
```

Finally, we look at the distribution of the target variable **next_day_fatigue_score** and notice that it is close to normal distribution. 


<Figure size 640x480 with 1 Axes><img width="572" height="433" alt="image" src="https://github.com/user-attachments/assets/2f85662f-51bc-481e-ba9d-c7aa0966391d" />

The computation of pairwise correlation between the target variable and each numerical features allow to eliminate the less instructive features such as **room_temperature_f**, **user_id** and **physical_activity_min**.

```python
df[numerical_variable].corrwith(df.next_day_fatigue_score).abs().sort_values(ascending=False)
```
```
total_sleep_hours             0.688568
sleep_efficiency_pct          0.536935
sleep_onset_latency_min       0.479315
awakenings_count              0.440525
rem_sleep_pct                 0.337327
screen_time_before_bed_min    0.336067
age                           0.278334
caffeine_intake_mg            0.261541
deep_sleep_pct                0.190759
screen_brightness_pct         0.176128
blue_light_filter_on          0.132120
bedtime_hour                  0.132016
caffeine_last_hour            0.131273
ambient_noise_db              0.067587
physical_activity_min         0.016990
user_id                       0.013405
room_temperature_f            0.011141
dtype: float64
```

A similar analysis on categorical variables shows that the only one that has significant effect on sleep score is **chronotype**, then we eliminate the others.

## Model training

After splitting the dataset into **training**, **validation** and **test** with **train_test_split**, and transforming the numerical features with **StandardScaler** and the categorical feature with **OneHotEncoder**, we are now ready to train several model.

The first one we want to try is the linear regression which is the most simple model for a regression problem.

The root mean squared error for this model is 0.418.
Then, we try Ridge and Lasso regression, and tune their * * alpha * * parameter. The best tunnings for these two models give exactly the same result as the linear regression but the avoid the high coefficients of the model.

Then we try **DecisionTreeRegressor** and **RandomForestRegressor**, but even when tunning their parameter the best root mean squared errors are respectively 0.537 and 0.463 so these two models are worse than the previous ones.
Same thing for **XGBoost** which give a room mean squared error of 0.470.

Finally we try a **KernelRidge** model that give a 0.413 root mean squared error with the following tuned parameters :
```python
degree =3
gamma = 0.003
alpha = 0.1
kernel = 'poly'
```

This model is interresting because it gives a slightly better result but at the cost of greater complexity. Indeed, since this model is non-linear, it is much more difficult to interpret.
Contrary to linear models, we can not rank the features impact on the target variable by looking at the model's coefficients.
Since the performance increase is not very important, we chose to keep a simple linear model.

Finally, we train the final model with the full training dataset (training + validation) : a Lasso regressor with alpha=0.0005.
On this new trainig dataset the root mean squared error is even better : 0.406, and the coefficients of the model, sorted by absolute values are :

```python
pd.DataFrame({'features':features, 'coef':las.coef_, 'abs(coef)':abs(las.coef_)}).sort_values(by='abs(coef)',ascending=False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>coef</th>
      <th>abs(coef)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>total_sleep_hours</td>
      <td>-0.600169</td>
      <td>0.600169</td>
    </tr>
    <tr>
      <th>14</th>
      <td>primary_app_category_E-Reading</td>
      <td>-0.315365</td>
      <td>0.315365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>awakenings_count</td>
      <td>0.244868</td>
      <td>0.244868</td>
    </tr>
    <tr>
      <th>18</th>
      <td>primary_app_category_Short-Form Video</td>
      <td>0.186620</td>
      <td>0.186620</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sleep_efficiency_pct</td>
      <td>-0.152630</td>
      <td>0.152630</td>
    </tr>
    <tr>
      <th>5</th>
      <td>screen_time_before_bed_min</td>
      <td>0.133989</td>
      <td>0.133989</td>
    </tr>
    <tr>
      <th>15</th>
      <td>primary_app_category_Gaming</td>
      <td>0.125909</td>
      <td>0.125909</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rem_sleep_pct</td>
      <td>-0.111924</td>
      <td>0.111924</td>
    </tr>
    <tr>
      <th>19</th>
      <td>primary_app_category_Social Media</td>
      <td>0.104756</td>
      <td>0.104756</td>
    </tr>
    <tr>
      <th>17</th>
      <td>primary_app_category_News</td>
      <td>-0.075915</td>
      <td>0.075915</td>
    </tr>
    <tr>
      <th>9</th>
      <td>screen_brightness_pct</td>
      <td>0.072634</td>
      <td>0.072634</td>
    </tr>
    <tr>
      <th>10</th>
      <td>blue_light_filter_on</td>
      <td>-0.070845</td>
      <td>0.070845</td>
    </tr>
    <tr>
      <th>7</th>
      <td>caffeine_intake_mg</td>
      <td>0.053260</td>
      <td>0.053260</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sleep_onset_latency_min</td>
      <td>0.044667</td>
      <td>0.044667</td>
    </tr>
    <tr>
      <th>20</th>
      <td>primary_app_category_Streaming</td>
      <td>-0.038833</td>
      <td>0.038833</td>
    </tr>
    <tr>
      <th>12</th>
      <td>caffeine_last_hour</td>
      <td>0.032420</td>
      <td>0.032420</td>
    </tr>
    <tr>
      <th>11</th>
      <td>bedtime_hour</td>
      <td>0.015942</td>
      <td>0.015942</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ambient_noise_db</td>
      <td>-0.009966</td>
      <td>0.009966</td>
    </tr>
    <tr>
      <th>8</th>
      <td>deep_sleep_pct</td>
      <td>-0.005654</td>
      <td>0.005654</td>
    </tr>
    <tr>
      <th>6</th>
      <td>age</td>
      <td>0.003716</td>
      <td>0.003716</td>
    </tr>
    <tr>
      <th>16</th>
      <td>primary_app_category_Messaging</td>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>



 ## Deployment

