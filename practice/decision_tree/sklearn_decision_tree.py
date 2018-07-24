#!/usr/bin/python

# https://www.ritchieng.com/machine-learning-decision-trees

import csv
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics


data_csv_file = "data.csv"

# Read all data into memory
raw_data = []
with open(data_csv_file) as csvfile:
    csv_reader = csv.reader(csvfile)
    data_header = next(csv_reader)
    for row in csv_reader:
        raw_data.append(row)

# Make to be float value from string,
data = [[float(x) for x in row] for row in raw_data]

# Pull out feature variables
x_vals = np.array([x[:-1] for x in data])
# Pull out target variables
y_vals = np.array([x[-1] for x in data])

###############################################################################

# Split data, 4:1
x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.2, random_state=None)

# 1. Instantiate with min_samples_split = 4
dtc = DecisionTreeClassifier(min_samples_split=4, random_state=None)

# 2. Fit
dtc.fit(x_train, y_train)

# 3. Predict, there're 4 features in the iris dataset
y_pred_class = dtc.predict(x_test)

# 4. Accuracy
accuracy_score = metrics.accuracy_score(y_test, y_pred_class)
print("Accuracy scroe: {:%}".format(accuracy_score))

print y_pred_class
print y_test
