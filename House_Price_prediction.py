# write a python program to draw the 3d plot for the model developed for house price prediction using suitable python based 3d plotting libraries

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# ----------------------------
# Sample House Dataset
# ----------------------------
# Area (sqft)
area = np.array([1000, 1200, 1500, 1800, 2000,
                 2200, 2500, 2800, 3000, 3500])

# Bedrooms
bedrooms = np.array([2, 2, 3, 3, 3,
                     4, 4, 4, 5, 5])

# House Price (Lakhs)
price = np.array([30, 35, 45, 50, 55,
                  65, 70, 80, 90, 105])

# Input Features
X = np.column_stack((area, bedrooms))
y = price

# ----------------------------
# Train Model
# ----------------------------
model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Create Mesh Grid
# ----------------------------
area_range = np.linspace(area.min(), area.max(), 30)
bedroom_range = np.linspace(bedrooms.min(), bedrooms.max(), 30)

A, B = np.meshgrid(area_range, bedroom_range)

# Predict prices over mesh
Z = model.predict(
    np.column_stack((A.ravel(), B.ravel()))
).reshape(A.shape)

# ----------------------------
# Plot
# ----------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual Data Points
ax.scatter(
    area,
    bedrooms,
    price,
    color='red',
    s=80,
    label='Actual Houses'
)

# Prediction Surface
ax.plot_surface(
    A,
    B,
    Z,
    cmap='viridis',
    alpha=0.7
)

ax.set_title("House Price Prediction Model (3D)")
ax.set_xlabel("Area (sq ft)")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price (Lakhs)")

ax.legend()

plt.show()
