# read csv file
import pandas as pd
data = pd.read_csv('AirPassengers.csv')
# data
data['Date'] = pd.to_datetime(data['Date'])
import numpy as np
# extract month and year from dates**
data['Month'] = [i.month for i in data['Date']]
data['Year'] = [i.year for i in data['Date']]

# create a sequence of numbers
data['Series'] = np.arange(1,len(data)+1)
# split data into train-test set
train = data[data['Year'] < 1953]
test = data[data['Year'] >= 1953]

# import the regression module**
from pycaret.regression import *
# initialize setup**
s = setup(data = train, test_data = test, target = '#Passengers', fold_strategy = 'timeseries', numeric_features = ['Year', 'Series'], fold = 3, transform_target = True, session_id = 123)

best = compare_models(sort = 'MAE')
print("In ra modle tot nhat: ", best)
prediction_holdout = predict_model(best)
print("In ra prediction_holdout:", prediction_holdout)
predictions = predict_model(best, data=data)
print("In ra:", predictions)