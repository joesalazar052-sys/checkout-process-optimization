# checkout-process-optimization
Python code
1.1 : Introduce the problem statement you are focusing on. Explain why it is of interest.
Is there a relationship between age, weight, or height in terms of medals won and if so which sports?

It is of interest because it shows the contrast between different athletes and if there is an advantage to selecting athletes that fall under certain requirements such as age, weight, or height. It is of interest because it allows countries to make better selections of athletes depending on their physical capabilities.
1.2: Briefly outline your approach to solving this problem, including the data and methodology you will use.
We will begin by analyzing the data. We discovered that there were multiple athlete duplicates. By removing these extras it will prevent the data from being skewed. We cleaned and organized the data by first removing any variables that are not in the scope of our study, which was a vast majority of them. Afterwards, we made sure to add a value if no medal was received, as they originally left the datapoint blank. We then proceeded to make boxplots and graphs to see if there are any correlations between the variables and medals won. We finished with a logistic regression to see if the variables are actually significant to medals won.
1.3 : Describe the analytical techniques you propose to use to address the problem, either fully or partially.
First, to gain a better understanding of the data we plan on using boxplots to look at potential outliers and to remove anything that could potentially cause skews in the data. From there, we will also use barcharts to see which countries have the highest number of medals, which we can then use to look at the histograms of height and weight to check to see what the average distribution is and if there is a consistent weight/height people are for the Olympics.
1.4: Clarify how your analysis will benefit the end user.
The analysis will benefit the end user by allowing countries to see if there is a more efficient way to select athletes for competitions. While many people apply for the Olympics and have to run through multiple tests to prove their ability to be on the team, this could help potentially narrow down the amount of people who go through initial testing and save time to focus more on competition preparation for the people actually competing.
Section 2
2.1: Load all necessary Python libraries at the beginning so that the reader knows what is needed for replication.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
2.2: Suppress any messages or warnings that arise from loading libraries.
warnings.filterwarnings('ignore')
2.3: Provide explanations for why each library is used.
We used pandas, seaborn, as well as matplotlib.pyplot data manipulation and analysis. Pandas is used since it allows the easy access of dataframes and data manipulation, allowing us to very easily clean the data and start understanding it in a deeper way. The reason for using the other libraries (matplotlib, pyplot, statsmodels, numpy, warnings, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, and seaborn) is that they allow us to visualize the data in a meaningful and easy way for people who may not have as much experience in statistics.
Section 3
3.1 Cite and hyperlink the original source of the data, if possible.
https://github.com/rfordatascience/tidytuesday/blob/main/data/2024/2024-08-06/readme.md

Harmon, J. (2024, August 6). Tidytuesday/Data/2024/2024-08-06/readme.md at main · rfordatascience/tidytuesday. GitHub. https://github.com/rfordatascience/tidytuesday/blob/main/data/2024/2024-08-06/readme.md
3.2: Offer a comprehensive explanation of the source data, including its original purpose, collection date, and any peculiarities like missing values or imputed data.
The source data comes from www.sports-reference.com, and was scraped from the website in 2018 by Jon Harmon. On sports-reference, it is a massive collection of data from all facets of sports, with the purpose of keeping the history of all sports and the significance it has had on our society. The data has been collected since 1896 in Athens, and has been continuously updated since then. It is important to note that after 1992 they began to stagger the Winter and Summer games into 2 year gaps, but beforehand all of the sports were held in the same year.
3.3: Explain the steps taken for data importing and cleaning, and justify your actions.
We downloaded the zip Olympics file. Unzipped the CSV file and downloaded it to JupyterLab using the read.csv and pandas. There were also quite a few duplicate athletes along with certain teams having multiple but under one country. We also renamed countries to allow for uniform understanding of specific countries that had multiple names. By removing these additional athletes it ensures the data is cleaner, it is easier to filter through, and that we can manipulate it efficiently. We also removed any variables that aren’t being used in the analysis we are conducting to allow for an easier viewing of the data. Another cleaning step we had to take was to fill in the values for if someone didn’t receive a medal, as they were left empty in the original dataset.
3.4 Show the final cleaned data set in a condensed form (just few rows).
olympics = pd.read_csv('olympics.csv')
olympics.head()
olympics = olympics.drop_duplicates().reset_index(drop = True)
olympics = olympics.drop(['sex', 'noc', 'games', 'year', 'city', 'event'], axis = 1)
olympics['medal'].fillna('No Medal', inplace=True)
olympics = olympics.dropna(subset=['age', 'height', 'weight'])
print(olympics.head())
   id                      name   age  height  weight         team  season  \
0   1                 A Dijiang  24.0   180.0    80.0        China  Summer   
1   2                  A Lamusi  23.0   170.0    60.0        China  Summer   
4   5  Christine Jacoba Aaftink  21.0   185.0    82.0  Netherlands  Winter   
5   5  Christine Jacoba Aaftink  21.0   185.0    82.0  Netherlands  Winter   
6   5  Christine Jacoba Aaftink  25.0   185.0    82.0  Netherlands  Winter   

           sport     medal  
0     Basketball  No Medal  
1           Judo  No Medal  
4  Speed Skating  No Medal  
5  Speed Skating  No Medal  
6  Speed Skating  No Medal  
3.5 : Summarize the key variables in your cleaned data set, either in a table or through a well-crafted summary paragraph with Python code. Explain your summary
n_rows      = len(olympics)
n_athletes  = olympics["name"].nunique()
n_teams     = olympics["team"].nunique()
n_sports    = olympics["sport"].nunique()
seasons     = ", ".join(sorted(olympics["season"].dropna().unique()))

avg_age     = olympics["age"].mean()
avg_height  = olympics["height"].mean()
avg_weight  = olympics["weight"].mean()

medal_counts = olympics["medal"].value_counts()
prop_no_medal = medal_counts.get("No Medal", 0) / n_rows
prop_medal = 1 - prop_no_medal

summary = (
    f"The cleaned athletes dataset contains {n_rows} rows, representing "
    f"{n_athletes} unique competitors from {n_teams} teams. "
    f"These athletes participate in {n_sports} sports across the {seasons} Games. "
    f"On average, athletes are {avg_age:.1f} years old, "
    f"{avg_height:.1f} cm tall, and weigh {avg_weight:.1f} kg. "
    f"Most observations are labeled as '{medal_counts.idxmax()}', and overall "
    f"{prop_medal:.1%} of records correspond to athletes who have won at least "
    f"one medal, while the rest are recorded as 'No Medal'."
)

print(summary)
The cleaned athletes dataset contains 206152 rows, representing 98545 unique competitors from 660 teams. These athletes participate in 56 sports across the Summer, Winter Games. On average, athletes are 25.1 years old, 175.4 cm tall, and weigh 70.7 kg. Most observations are labeled as 'No Medal', and overall 14.6% of records correspond to athletes who have won at least one medal, while the rest are recorded as 'No Medal'.
Section 4
4.1: Discuss what you found and how you tried to reveal hidden insights in the data.
When looking at the data, we started to realize that all of the winners between gold, silver, and bronze are quite similar in weight, height and age when compared to the non-medalists. All of the boxplots show that all of the means are similar, and while there is slight variation, it is not enough to see any sort of significance. It is important to note there are a LOT of outliers within the data, and that the data itself may not be the most reliable due to the amount of missing data and things we had to standardize.
4.2: Use proper types of plots and tables that help illustrate your findings. Please limit your figures and tables totaled at 8 , and make sure you have an explanation after each plot or table.
sns.boxplot(x='medal', y='age', data= olympics)
plt.xticks(rotation = 45, ha = 'right')
plt.title('Medals by Age')
plt.show()
No description has been provided for this image
Athletes across all the different medal groups tend to be in their mid 20s, and the age differences between medalist and non-medalist are small. Gold and Bronze medalists are slightly older on average, but not by a lot. We also see a wide age range, from teens to 40s and older, showing that people can stay competitive in sports for a long time. This graph shows that age does not seem to factor into medal success.
sns.boxplot(x='medal', y='weight', data= olympics)
plt.xticks(rotation = 45, ha = 'right')
plt.title('Medals by weight')
plt.show()
No description has been provided for this image
The box plot shows that medal winners and non-medal winners are very similar in weight ranges. This suggests that weight does not influence medal success. All four have just about the same median weight. This graph strongly suggests that there is no correlation with weight and medals.
sns.boxplot(x='medal', y='height', data= olympics)
plt.xticks(rotation = 45, ha = 'right')
plt.title('Medals by Height')
plt.show()
No description has been provided for this image
The box plot shows that height does not have a strong influence on whether an athlete wins a medal. The median and range of height is about the same across all classes of medals.
Overall there does not seem to be any correlation between age, weight, or height that influences the medals won.
4.3: Use newly learned skills such as functions to handle character variables, or use interactions and self-made functions to show more insights you discovered from the dataset.
high_age = olympics.groupby('medal')['age'].max()
print("High Age: \n", high_age)

low_age = olympics.groupby('medal')['age'].min()
print("Low Age: \n", low_age)

high_weight = olympics.groupby('medal')['weight'].max()
print("High Weight: \n", high_weight)

low_weight = olympics.groupby('medal')['weight'].min()
print("Low Weight: \n", low_weight)

high_height = olympics.groupby('medal')['height'].max()
print("High height: \n", high_height)

low_height = olympics.groupby('medal')['height'].min()
print("Low Height: \n", low_height)
High Age: 
 medal
Bronze      61.0
Gold        59.0
No Medal    71.0
Silver      66.0
Name: age, dtype: float64
Low Age: 
 medal
Bronze      13.0
Gold        13.0
No Medal    11.0
Silver      13.0
Name: age, dtype: float64
High Weight: 
 medal
Bronze      182.0
Gold        170.0
No Medal    214.0
Silver      167.0
Name: weight, dtype: float64
Low Weight: 
 medal
Bronze      28.0
Gold        28.0
No Medal    25.0
Silver      30.0
Name: weight, dtype: float64
High height: 
 medal
Bronze      223.0
Gold        223.0
No Medal    226.0
Silver      223.0
Name: height, dtype: float64
Low Height: 
 medal
Bronze      136.0
Gold        136.0
No Medal    127.0
Silver      136.0
Name: height, dtype: float64
Athletes who win medals tend to have more balanced heights and weights, while those who don't have medals show more extreme on both high and low ends. Hieght stands out the most with a steady and predictable range. Weight shows a similar pattern with non-medalists appearing at the most extreme values and medalist staying more centered. Age does not seem to make much difference, since all groups share almost the same age range. Overall, medalists semm to come from a more consistent physical profile, while non-medalists vary much more.
Section 5
5.1 State which statistical or machine learning models you used in your analysis.
We chose to use Logistic Regression.
5.2: Explain the rationale behind choosing the model. What specific questions do you aim to answer or insights do you aim to gain?
We had to use logistic regression because medals are categorical variables and incompatible with linear regression. The main goal to address with the model is to see if there is any difference between medal winners and non-medal winners. Since we have the resources and time, we were also curious to see if there is a difference between the specific medals and no medals. For example, if bronze winners tend to have a significant age, weight, height compared to silver or gold medalists.
5.3: Discuss the results you got from using the model.
#Create a dummy code for Medals
medal_dummy = pd.get_dummies(olympics['medal'], dtype=int)
medal_dummy

#Combine dummy code with original dataset
olympics_encoded = pd.concat([olympics, medal_dummy ], axis=1)
olympics_encoded.head()
id	name	age	height	weight	team	season	sport	medal	Bronze	Gold	No Medal	Silver
0	1	A Dijiang	24.0	180.0	80.0	China	Summer	Basketball	No Medal	0	0	1	0
1	2	A Lamusi	23.0	170.0	60.0	China	Summer	Judo	No Medal	0	0	1	0
4	5	Christine Jacoba Aaftink	21.0	185.0	82.0	Netherlands	Winter	Speed Skating	No Medal	0	0	1	0
5	5	Christine Jacoba Aaftink	21.0	185.0	82.0	Netherlands	Winter	Speed Skating	No Medal	0	0	1	0
6	5	Christine Jacoba Aaftink	25.0	185.0	82.0	Netherlands	Winter	Speed Skating	No Medal	0	0	1	0
# Select variables needed for the model
data = olympics_encoded[[ 'height', 'weight', 'Gold']].dropna()

# Prepare the features and target variable
X = data[['height','weight']]  # Using 'total_bill' as the dependent variable or predictor variable
X = sm.add_constant(X) # Adds a constant term to the predictor

# Use Gold dummy column as the dependent variable
y = data['Gold']

# Create a logistic regression model
model = sm.Logit(y, X)

# Fit the model
Gold = model.fit()

# Display the model summary for evaluation
print(Gold.summary())

# Extracting coefficients and calculating odds ratios
coefficients_gold = Gold.params
odds_ratios_gold = np.exp(coefficients_gold)

print("Gold Coefficients: \n", coefficients_gold)
print("Gold Odds Ratios: \n", odds_ratios_gold)

# Predict probabilities
predicted_probabilities_gold = Gold.predict(X)
print("Gold Predicted Probabilities:")
print(predicted_probabilities_gold.head())

# Convert probabilities to class labels
predicted_classes_gold = (predicted_probabilities_gold > 0.5).astype(int)
print("Gold Predicted Classes:")
print(predicted_classes_gold.head())

# Drop missing values
predicted_classes_gold.dropna()

# first argument - the true classes, second argument - predicted classes
accuracy_gold = accuracy_score(y, predicted_classes_gold)

# Calculate evaluation metrics
precision_gold = precision_score(y, predicted_classes_gold)
recall_gold = recall_score(y, predicted_classes_gold)

print(f"Accuracy: {accuracy_gold}")
print(f"Precision: {precision_gold}")
print(f"Recall: {recall_gold}")
print(f"Accuracy: {accuracy_gold}")

# ROC and AUC for Gold
fpr, tpr, thresholds = roc_curve(y, predicted_probabilities_gold)  # Assuming y_pred_prob is the predicted probability
auc = roc_auc_score(y, predicted_probabilities_gold)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Gold Medals')
plt.legend(loc='best')
plt.show()
Optimization terminated successfully.
         Current function value: 0.194790
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                   Gold   No. Observations:               206152
Model:                          Logit   Df Residuals:                   206149
Method:                           MLE   Df Model:                            2
Date:                Sat, 22 Nov 2025   Pseudo R-squ.:                0.008712
Time:                        13:15:54   Log-Likelihood:                -40156.
converged:                       True   LL-Null:                       -40509.
Covariance Type:            nonrobust   LLR p-value:                5.427e-154
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -6.4651      0.217    -29.784      0.000      -6.890      -6.040
height         0.0169      0.002     11.042      0.000       0.014       0.020
weight         0.0072      0.001      6.698      0.000       0.005       0.009
==============================================================================
Gold Coefficients: 
 const    -6.465051
height    0.016919
weight    0.007167
dtype: float64
Gold Odds Ratios: 
 const     0.001557
height    1.017063
weight    1.007192
dtype: float64
Gold Predicted Probabilities:
0    0.054870
1    0.040743
4    0.060232
5    0.060232
6    0.060232
dtype: float64
Gold Predicted Classes:
0    0
1    0
4    0
5    0
6    0
dtype: int32
Accuracy: 0.9506820210330241
Precision: 0.0
Recall: 0.0
Accuracy: 0.9506820210330241
No description has been provided for this image
This logistic model is trying to predict the likelihood of winning a gold medal using height and weight as predictors. The p-value for this model is smaller than 0.05, meaning it is statistically significant. Both height and weight have a smaller p-value smaller than 0.05, so statistically they are both good predictors. The coefficient is at -6.465051, showing that the likelihood of obtaining a gold medal occurring is very low when both predictors are at zero.¶

According to the odds ratio, both height and weight have a small positive effect so if an athlete height were to increase by 1 unit, then there is a 1.7% chance of that athlete getting the gold medal and if there weight were to increase by 1 unit then the athlete would have a 0.7% chance of getting the gold medal.

The very high accuracy indicates the model was able to generally make accurate predictions, but with a very low precision and recall it had a high false positive rate and failed to capture most of the positive instances. This can be supported by the ROC graph, which shows an AUC of 0.57.
# Select variables needed for the model
data = olympics_encoded[[ 'height', 'weight', 'Silver']].dropna()

# Prepare the features and target variable
X = data[['height','weight']]  # Using 'total_bill' as the dependent variable or predictor variable
X = sm.add_constant(X) # Adds a constant term to the predictor

# Use Silver dummy column as the dependent variable
y = data['Silver']

# Create a logistic regression model
model = sm.Logit(y, X)

# Fit the model
Silver = model.fit()

# Display the model summary for evaluation
print(Silver.summary())

# Extracting coefficients and calculating odds ratios
coefficients_silver = Silver.params
odds_ratios_silver = np.exp(coefficients_silver)

print("Silver Coefficients: \n", coefficients_silver)
print("Silver Odds Ratios: \n", odds_ratios_silver)

# Predict probabilities
predicted_probabilities_silver = Silver.predict(X)
print("Silver Predicted Probabilities:")
print(predicted_probabilities_silver.head())


# Convert probabilities to class labels
predicted_classes_silver = (predicted_probabilities_silver > 0.5).astype(int)
print("Silver Predicted Classes:")
print(predicted_classes_silver.head())

# Drop missing values
predicted_classes_silver.dropna()

# first argument - the true classes, second argument - predicted classes
accuracy_silver = accuracy_score(y, predicted_classes_silver)

# Calculate evaluation metrics
precision_silver = precision_score(y, predicted_classes_silver)
recall_silver = recall_score(y, predicted_classes_silver)

print(f"Accuracy: {accuracy_silver}")
print(f"Precision: {precision_silver}")
print(f"Recall: {recall_silver}")
print(f"Accuracy: {accuracy_silver}")

# ROC and AUC for Silver
fpr, tpr, thresholds = roc_curve(y, predicted_probabilities_silver)  # Assuming y_pred_prob is the predicted probability
auc = roc_auc_score(y, predicted_probabilities_silver)

plt.figure() 
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Silver Medals')
plt.legend(loc='best')
plt.show()
Optimization terminated successfully.
         Current function value: 0.190975
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 Silver   No. Observations:               206152
Model:                          Logit   Df Residuals:                   206149
Method:                           MLE   Df Model:                            2
Date:                Sat, 22 Nov 2025   Pseudo R-squ.:                0.006164
Time:                        13:14:16   Log-Likelihood:                -39370.
converged:                       True   LL-Null:                       -39614.
Covariance Type:            nonrobust   LLR p-value:                9.005e-107
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -5.8285      0.220    -26.464      0.000      -6.260      -5.397
height         0.0133      0.002      8.539      0.000       0.010       0.016
weight         0.0068      0.001      6.183      0.000       0.005       0.009
==============================================================================
Silver Coefficients: 
 const    -5.828511
height    0.013321
weight    0.006771
dtype: float64
Silver Odds Ratios: 
 const     0.002942
height    1.013411
weight    1.006794
dtype: float64
Silver Predicted Probabilities:
0    0.052701
1    0.040792
4    0.056849
5    0.056849
6    0.056849
dtype: float64
Silver Predicted Classes:
0    0
1    0
4    0
5    0
6    0
dtype: int32
Accuracy: 0.9521421087353021
Precision: 0.0
Recall: 0.0
Accuracy: 0.9521421087353021
No description has been provided for this image
This logistic model is trying to predict the likelihood of winning a silver medal using height and weight as predictors. The p-value for this model is smaller than 0.05, meaning it is statistically significant. Both height and weight have a smaller p-value smaller than 0.05, so statistically they are both good predictors. The coefficient is at -5.828511, showing that the likelihood of obtaining a silver medal occurring is very low when both predictors are at zero.

According to the odds ratio, both height and weight have a small positive effect so if an athlete height were to increase by 1 unit, then there is a 1.3% chance of that athlete getting the silver medal and if there weight were to increase by 1 unit then the athlete would have a 0.7% chance of getting the silver medal.

The very high accuracy indicates the model was able to generally make accurate predictions, but with a very low precision and recall it had a high false positive rate and failed to capture most of the positive instances. This can be supported by the ROC graph, which shows an AUC of 0.56.
# Select variables needed for the model
data = olympics_encoded[[ 'height', 'weight', 'Bronze']].dropna()

# Prepare the features and target variable
X = data[['height','weight']]  # Using 'total_bill' as the dependent variable or predictor variable
X = sm.add_constant(X) # Adds a constant term to the predictor

# Use Bronze dummy column as the dependent variable
y = data['Bronze']

# Create a logistic regression model
model = sm.Logit(y, X)

# Fit the model
Bronze = model.fit()

# Display the model summary for evaluation
print(Bronze.summary())


# Extracting coefficients and calculating odds ratios
coefficients_bronze = Bronze.params
odds_ratios_bronze = np.exp(coefficients_bronze)

print("Bronze Coefficients: \n", coefficients_bronze)
print("Bronze Odds Ratios: \n", odds_ratios_bronze)

# Predict probabilities
predicted_probabilities_bronze = Bronze.predict(X)
print("Bronze Predicted Probabilities:")
print(predicted_probabilities_bronze.head())


# Convert probabilities to class labels
predicted_classes_bronze = (predicted_probabilities_bronze > 0.5).astype(int)
print("Bronze Predicted Classes:")
print(predicted_classes_bronze.head())


# Drop missing values
predicted_classes_bronze.dropna()

# first argument - the true classes, second argument - predicted classes
accuracy_bronze = accuracy_score(y, predicted_classes_bronze)

# Calculate evaluation metrics
precision_bronze = precision_score(y, predicted_classes_bronze)
recall_bronze = recall_score(y, predicted_classes_bronze)

print(f"Accuracy: {accuracy_bronze}")
print(f"Precision: {precision_bronze}")
print(f"Recall: {recall_bronze}")
print(f"Accuracy: {accuracy_bronze}")


# ROC and AUC for Bronze
fpr, tpr, thresholds = roc_curve(y, predicted_probabilities_bronze)  # Assuming y_pred_prob is the predicted probability
auc = roc_auc_score(y, predicted_probabilities_bronze)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Bronze Medals')
plt.legend(loc='best')
plt.show()
Optimization terminated successfully.
         Current function value: 0.195185
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 Bronze   No. Observations:               206152
Model:                          Logit   Df Residuals:                   206149
Method:                           MLE   Df Model:                            2
Date:                Sat, 22 Nov 2025   Pseudo R-squ.:                0.005320
Time:                        13:13:45   Log-Likelihood:                -40238.
converged:                       True   LL-Null:                       -40453.
Covariance Type:            nonrobust   LLR p-value:                 3.485e-94
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -5.2809      0.217    -24.353      0.000      -5.706      -4.856
height         0.0099      0.002      6.433      0.000       0.007       0.013
weight         0.0080      0.001      7.472      0.000       0.006       0.010
==============================================================================
Bronze Coefficients: 
 const    -5.280865
height    0.009881
weight    0.008028
dtype: float64
Bronze Odds Ratios: 
 const     0.005088
height    1.009929
weight    1.008061
dtype: float64
Bronze Predicted Probabilities:
0    0.054161
1    0.042311
4    0.057614
5    0.057614
6    0.057614
dtype: float64
Bronze Predicted Classes:
0    0
1    0
4    0
5    0
6    0
dtype: int32
Accuracy: 0.9507741860374869
Precision: 0.0
Recall: 0.0
Accuracy: 0.9507741860374869
No description has been provided for this image
This logistic model is trying to predict the likelihood of winning a bronze medal using height and weight as predictors. The p-value for this model is smaller than 0.05, meaning it is statistically significant. Both height and weight have a smaller p-value smaller than 0.05, so statistically they are both good predictors. The coefficient is at -5.280865, showing that the likelihood of obtaining a bronze medal occurring is very low when both predictors are at zero.

According to the odds ratio, both height and weight have a small positive effect so if an athlete height were to increase by 1 unit, then there is a 0.10% chance of that athlete getting the bronze medal and if there weight were to increase by 1 unit then the athlete would have a 0.8% chance of getting the medal.

The very high accuracy indicates the model was able to generally make accurate predictions, but with a very low precision and recall it had a high false positive rate and failed to capture most of the positive instances. This can be supported by the ROC graph, which shows an AUC of 0.56.
# Select variables needed for the model
data = olympics_encoded[[ 'height', 'weight', 'No Medal']].dropna()

# Prepare the features and target variable
X = data[['height','weight']]  # Using 'total_bill' as the dependent variable or predictor variable
X = sm.add_constant(X) # Adds a constant term to the predictor

# Use No Medal dummy column as the dependent variable
y = data['No Medal']

# Create a logistic regression model
model = sm.Logit(y, X)

# Fit the model
no_medal = model.fit()

# Display the model summary for evaluation
print(no_medal.summary())

# Extracting coefficients and calculating odds ratios
coefficients_no_medal = no_medal.params
odds_ratios_no_medal = np.exp(coefficients_no_medal)

print("No Medals Coefficients: \n", coefficients_no_medal)
print("No Medals Odds Ratios: \n", odds_ratios_no_medal)

# Predict probabilities
predicted_probabilities_no_medal = no_medal.predict(X)
print("No Medals Predicted Probabilities:")
print(predicted_probabilities_no_medal.head())

# Convert probabilities to class labels
predicted_classes_no_medal = (predicted_probabilities_no_medal > 0.5).astype(int)
print("No Medals Predicted Classes:")
print(predicted_classes_no_medal.head())

# Drop missing values
predicted_classes_no_medal.dropna()

# first argument - the true classes, second argument - predicted classes
accuracy_no_medal = accuracy_score(y, predicted_classes_no_medal)

# Calculate evaluation metrics
precision_no_medal = precision_score(y, predicted_classes_no_medal)
recall_no_medal = recall_score(y, predicted_classes_no_medal)

print(f"Accuracy: {accuracy_no_medal}")
print(f"Precision: {precision_no_medal}")
print(f"Recall: {recall_no_medal}")
print(f"Accuracy: {accuracy_no_medal}")

# ROC and AUC for No Medal
fpr, tpr, thresholds = roc_curve(y, predicted_probabilities_no_medal)  # Assuming y_pred_prob is the predicted probability
auc = roc_auc_score(y, predicted_probabilities_no_medal)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for No Medals')
plt.legend(loc='best')
plt.show()
Optimization terminated successfully.
         Current function value: 0.412061
         Iterations 6
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               No Medal   No. Observations:               206152
Model:                          Logit   Df Residuals:                   206149
Method:                           MLE   Df Model:                            2
Date:                Sat, 22 Nov 2025   Pseudo R-squ.:                 0.01046
Time:                        13:13:10   Log-Likelihood:                -84947.
converged:                       True   LL-Null:                       -85845.
Covariance Type:            nonrobust   LLR p-value:                     0.000
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          4.9817      0.134     37.140      0.000       4.719       5.245
height        -0.0148      0.001    -15.573      0.000      -0.017      -0.013
weight        -0.0084      0.001    -12.509      0.000      -0.010      -0.007
==============================================================================
No Medals Coefficients: 
 const     4.981662
height   -0.014826
weight   -0.008405
dtype: float64
No Medals Odds Ratios: 
 const     145.716403
height      0.985283
weight      0.991630
dtype: float64
No Medals Predicted Probabilities:
0    0.837608
1    0.876197
4    0.824855
5    0.824855
6    0.824855
dtype: float64
No Medals Predicted Classes:
0    1
1    1
4    1
5    1
6    1
dtype: int32
Accuracy: 0.8535983158058131
Precision: 0.8535983158058131
Recall: 1.0
Accuracy: 0.8535983158058131
No description has been provided for this image
This logistic model is trying to predict the likelihood of not winning a medal using height and weight as predictors. The p-value for this model is smaller than 0.05, meaning it is statistically significant. Both height and weight have a smaller p-value smaller than 0.05, so statistically they are both good predictors. The coefficient is at 4.981662, showing that the likelihood of not obtaining a medal occurring is somewhat high when both predictors are at zero.

According to the odds ratio, both height and weight have a large positive effect so if an athlete height were to increase by 1 unit, then there is a 98.53% chance of that athlete getting no medal and if there weight were to increase by 1 unit then the athlete would have a 99.16% chance of getting no medal.

The very high accuracy indicates the model was able to generally make accurate predictions, and with a high precision and recall it had a low false positive rate and was able to capture all of the positive instances. However, the ROC graph shows an AUC of 0.56.
