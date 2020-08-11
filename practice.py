import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split


#data is from the men’s professional tennis league, which is called the ATP (Association of Tennis Professionals). Data from the top 1500 ranked players in the ATP over the span of 2009 to 2017 are provided in file. 
data = pd.read_csv('./tennis_stats.csv') 
print(data.head()) 

#for easiness with understanding data, I always print out column names to know exactly what is provided by the dataset 
columns_list = data.columns.tolist() 
columns_list_float = columns_list.remove('Player') 
#['Player', 'Year', 'FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon', 'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted', 'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon', 'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalPointsWon', 'TotalServicePointsWon', 'Wins', 'Losses', 'Winnings', 'Ranking']

'''Observing data structure: 
- general info 
- offensive rounds 
- defensive rounds
- outcomes 
'''

#splitting data into structure 

#exploratory analysis - plotting random features against outcome to see what has a correlation 
'''
plt.scatter(data['BreakPointsOpportunities'], data['Wins']) #strong correlation  - Focus 1 
plt.scatter(data['DoubleFaults'], data['Losses']) #some correlation - Focus 2 
plt.scatter(data['ReturnGamesWon'], data['Ranking']) #ugly correlation, ignored '''

## ONE FEATURE LINEAR REGRESSION 

#--------- BREAK POINTS OPPORTUNITIES VS WINS 

#defining feature and outcome 
features = data[['BreakPointsOpportunities']]
outcome = data [['Wins']]

#splitting data into training and test sets to ensure validity after training 
#sklearn has a built in function train_test_split that splits the dataset 
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8) 

#fitting model 

regr = LinearRegression() #declaring regr as linear regression model 
regr.fit(features_train, outcome_train) #fitting training model 
regr.score(features_test, outcome_test) #scoring test model 
y_predicts = regr.predict(features_train) #get y-values for trained dataset 
predictions = regr.predict(features_test) #getting y-values for test dataset 

sns.set() 
plt.figure()
plt.scatter(features, outcome) 
plt.plot(features_train, y_predicts, color='green') #two lines plotted the scatter of features and outcomes, their linear regression model 
plt.xlabel('# of Break Point Opportunities')
plt.ylabel('# of Wins')
plt.title('Wins vs. Break Point Opportunities - Fitting Regression Line to Model')

sns.set() 
plt.figure()
plt.scatter(outcome_test, predictions) #plotting the test oucome to the predicted outcome (showing degree of correlation)
plt.xlabel('Wins - Actual Outcome')
plt.ylabel('Wins - Predicted Outcome') 
plt.title('Actual Wins vs Predicted Wins')

#--------REGRESSION MODEL FOR DOUBLE FAULTS AND LOSSES 

features1 = data[['DoubleFaults']] 
outcome1 = data[['Losses']]

features1_train, features1_test, outcome1_train, outcome1_test = train_test_split(features1, outcome1, train_size = 0.8) #once again, splitting model to train and test sets 

regr1 = LinearRegression() 
regr1.fit(features1_train, outcome1_train) 
regr.score(features1_test, outcome1_test) 
y_predict1 = regr1.predict(features1_train) 
predictions1 = regr1.predict(features1_test) 

sns.set() 
plt.figure()
plt.scatter(features1, outcome1)
plt.plot(features1_train, y_predict1) #plotting linear regression line 
plt.xlabel('# of Double Faults')
plt.ylabel('# of Losses') 
plt.title('# of Double Feaults vs # of Losses - Fitting Regression Line to Model') 

sns.set() 
plt.figure()
plt.scatter(outcome1_test, predictions1) #plotting actual outcomes vs predictions 
plt.xlabel('Losses - Actual Outcome')
plt.ylabel('Losses - Predicted Outcome') 
plt.title('Actual Losses vs Predicted Losses')
plt.show() 

## MULTIPLE FEATURES LINEAR REGRESSION - MULTIVARIATE REGRESSION 

## -----------ALL POSSIBLE WIN INDICATORS 
features2 = data[['FirstServePointsWon', 'FirstServeReturnPointsWon', 'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted', 'BreakPointsFaced', 'BreakPointsOpportunities']]
outcome2 = data['Wins']

#continue splitting data like before 
features2_train, features2_test, outcome2_train, outcome2_test = train_test_split(features2, outcome2, train_size = 0.8) 

#train model 
regr3 = LinearRegression() 
regr3.fit(features2_train, outcome2_train) #train algorithm
regr3.score(features2_test, outcome2_test) #scoreing algorithm 

predictions2 = regr3.predict(features2_test) 

sns.set() 
plt.figure()
plt.scatter(outcome2_test, predictions2) 
plt.xlabel('Wins - Actual Outcome')
plt.ylabel('Wins - Predicted Outcome') 
plt.title('Actual Wins vs Predicted Wins')
plt.show() 
