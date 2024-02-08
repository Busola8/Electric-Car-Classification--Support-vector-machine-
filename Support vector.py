# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:38:24 2024

@author: busola
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import math
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Electric_cars_dataset.csv")
description = data.describe()

data.nunique()
unique_electric = data["Electric Vehicle Type"].unique()
CAFV = data["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].unique()
city = data["City"].unique()
country = data["County"].unique()
modelyear = data["Model Year"].unique()
model = data["Model"].unique()
make = data["Make"].unique()
Electric_vehicle = data["Electric Vehicle Type"].unique() 
electric_utility = data["Electric Utility"].unique()
state = data["State"].unique()
vehicle_location = data["Vehicle Location"].unique()
vehicle_id = data["DOL Vehicle ID"].unique()
zip_code = data["ZIP Code"].unique()
expected_price = data["Expected Price ($1k)"].unique()
missing1 = data.isnull().sum()


data["Expected Price ($1k)"] = data["Expected Price ($1k)"].replace({"N/":np.nan})

imputer = SimpleImputer(strategy = "most_frequent")
new_data = imputer.fit_transform(data)
new_data = pd.DataFrame(new_data, columns = imputer.feature_names_in_)
missing2 = new_data.isnull().sum()

# You convert to numeric
new_data["Expected Price ($1k)"] = pd.to_numeric(new_data["Expected Price ($1k)"])
new_data.info()
checknull = new_data.isnull().sum()

new_data = new_data.drop(["ID", "VIN (1-10)", "County", "Model", "City", "DOL Vehicle ID", "ZIP Code", "Vehicle Location"], axis = 1)
new_data = pd.get_dummies(new_data, drop_first = True)

#scaling of data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_data)
scaled_data = pd.DataFrame(scaled_data, columns = scaler.feature_names_in_)


y = scaled_data["Expected Price ($1k)"]
x = scaled_data.drop("Expected Price ($1k)", axis = 1)

# selector = RFE(estimator = SVR(), n_features_to_select = 5)
# #feature importance
# new_x = pd.DataFrame(selector.fit_transform(x, y), columns = selector.get_feature_names_out())
# # Get feature importances
# feature_importances = selector.feature_importances_
# # Creating a DataFrame or a Series
# feature_importances_df = pd.DataFrame({'Feature': list(range(1, len(feature_importances)+1)),
        #                                'Importance': feature_importances})



x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 10)
regressor = SVR()
model = regressor.fit(x_train,y_train)

y_train_pred = model.predict(x_train)
y_pred = model.predict(x_test)
    

mse_train = mean_squared_error(y_train,y_train_pred)
print("RMSE_train: ",math.sqrt(mse_train))
print("r_squared_train: ",metrics.r2_score(y_train, y_train_pred))

mse_test = mean_squared_error(y_test, y_pred)
print("RMSE_test: ",math.sqrt(mse_test))
print("r_squared_test: ",metrics.r2_score(y_test, y_pred))

 
