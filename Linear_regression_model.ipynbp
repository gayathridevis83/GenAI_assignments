# 1. Can you build a multivariate linear regression model that can predict the product sales based on the advertising budget allocated to different channels. The features are TV Budget ($), Radio Budget ($), Newspaper Budget ($) and the output is Sales (units)
# The dataset is given below
# TV Budget ($),Radio Budget ($),Newspaper Budget ($),Sales (units)
# 230.1,37.8,69.2,22.1
# 44.5,39.3,45.1,10.4
# 17.2,45.9,69.3,9.3
# 151.5,41.3,58.5,18.5
# 180.8,10.8,58.4,12.9
# 8.7,48.9,75.0,7.2
# 57.5,32.8,23.5,11.8
# 120.2,19.6,11.6,13.2
# 144.1,16.0,40.3,15.6
# 111.6,12.6,37.9,12.2

#Solution:

# importing python libraries
import pandas as pd  
from sklearn.linear_model import LinearRegression 

# creating dataset for the given data
data = {
    "TV": [230.1,44.5,17.2,151.5,180.8,8.7,57.5,120.2,144.1,111.6],
    "Radio": [37.8,39.3,45.9,41.3,10.8,48.9,32.8,19.6,16.0,12.6],
    "Newspaper": [69.2,45.1,69.3,58.5,58.4,75.0,23.5,11.6,40.3,37.9],
    "Sales": [22.1,10.4,9.3,18.5,12.9,7.2,11.8,13.2,15.6,12.2]
}

df = pd.DataFrame(data) 

# Features and target  x axix is media communication data frame and y axis is sales data frame
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Model train the modem with input data.
model = LinearRegression()
model.fit(X, y) 

# Prediction example
sample = [[100, 20, 30]]
pred = model.predict(sample)

print("Predicted Sales:", pred[0])
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

