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
missing1 = data.isnull().sum()


y = data["Expected Price ($1k)"]
x = data.drop("Expected Price ($1k)", axis = 1)

selector = RFE(estimator = SVR(), n_features_to_select = 12)
#feature importance
new_x = pd.DataFrame(selector.fit(x, y), columns = selector.get_feature_names_out())


# imputer = SimpleImputer(strategy = "median")
# new_data = imputer.fit_transform(data[[""]])
# new_data = pd.DataFrame(new_data, columns = imputer.feature_names_in_)
# missing2 = new_data.isnull().sum()

# # selector = RFE(estimator = RandomForestClassifier(random_state = 0, verbose = 1), n_features_to_select = 2)
# # feature importance
# # new_x = pd.DataFrame(selector.fit_transform(x, y), columns = selector.get_feature_names_out())

# # You convert to numeric
# data["Expected Price ($1k)"] = pd.to_numeric(data["Expected Price ($1k)"], errors = "coerce")
# data.info()

# data = data.drop(data[["ID", "VIN (1-10)", "DOL Vehicle ID", "Vehicle Location"]], axis = 1)

# # data = pd.get_dummies(data, drop_first = True)

# new_data = imputer.fit_transform(data)
# new_data = pd.DataFrame(new_data, columns = imputer.feature_names_in_)
# missing = new_data.isnull().sum()

# y = new_data["Expected Price ($1k)"]
# x = new_data.drop("Expected Price ($1k)", axis = 1)

# selector = RFE(estimator = SVR(), n_features_to_select = 12)
# #feature importance
# new_x = pd.DataFrame(selector.fit_transform(x, y), columns = selector.get_feature_names_out())
 

# # x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 10)
# # regressor = SVR()
# # model = regressor.fit(x_train,y_train)

# # y_train_pred = model.predict(x_train)
# # y_pred = model.predict(x_test)
    

# # mse_train = mean_squared_error(y_train,y_train_pred)
# # print("RMSE: ",math.sqrt(mse_train))
# # print("r_squared: ",metrics.r2_score(y_train, y_train_pred))

# # mse_test = mean_squared_error(y_test, y_pred)
# # print("RMSE: ",math.sqrt(mse_test))
# # print("r_squared: ",metrics.r2_score(y_test, y_pred))

 
