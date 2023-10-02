import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/useless1801/Data_Customer/main/Data_Startups.csv'
df = pd.read_csv(url)
print(df)
dataset = df.drop (['City'], axis = 1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

X

y

from sklearn.impute import SimpleImputer
imputa = SimpleImputer(missing_values= np.nan, strategy='mean')
imputa.fit(X[:, 0:3])

X[:, 0:3] = imputa.transform(X[:, 0:3])

print(X)

print(y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create a label encoder for the categorical column (index 3)
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

# Specify the column(s) to one-hot encode, in this case, column 3 (index 3)
# You can add more columns to this list if needed.
columns_to_encode = [2]

# Create a column transformer
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), columns_to_encode)
    ],
    remainder='passthrough'  # Keep the remaining columns as is
)

X_encoded = ct.fit_transform(X)

X.astype(int)
y.astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit (X_train, y_train)

y_pred = regressor.predict (X_test)
y_pred

y_test

import matplotlib.pyplot as plt

# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values in Multiple Linear Regression")
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot in Multiple Linear Regression")
plt.axhline(y=0, color='k', linestyle='--')  # Adding a horizontal line at y=0
plt.show()

from sklearn import linear_model
model = linear_model.LinearRegression()

model.fit(X_train, y_train) #Indep variables, dep. variable to be predicted
print(model.score(X_train, y_train))  #Prints the R^2 value, a measure of how well

prediction_test = model.predict(X_test)
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)

import pickle

pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl','rb'))

model.predict(X_test)