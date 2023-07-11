import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Importing necessary libraries

# Upload the 'googleplaystore.csv' file
# Make sure to place the file in the same directory as this code file
# In the actual execution environment, the file upload process may differ

df = pd.read_csv('googleplaystore.csv')  # Read the CSV file into a DataFrame

# Preprocess the 'Size' column
df['Size'] = df['Size'].replace('Varies with device', np.nan)  # Replace 'Varies with device' with NaN
df['Size'] = df['Size'].str.replace('M', '').str.replace('k', '').str.replace(',', '')  # Remove characters 'M', 'k', and ','
df['Size'] = df['Size'].str.replace('+', '', regex=False).astype(float)  # Remove '+' and convert to float
df['Size'] = df['Size'].apply(lambda x: float(x) / 1024 if x >= 1024 else x)  # Convert size to MB if it is in GB

# Remove non-numeric characters from the 'Installs' column
df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '')

# Handle the value 'Free' in the 'Installs' column
df['Installs'] = df['Installs'].replace('Free', '0')

# Convert the 'Installs' column to integer
df['Installs'] = df['Installs'].astype(int)

# Handle the value 'Everyone' in the 'Price' column
df['Price'] = df['Price'].replace('Everyone', '0')

# Remove dollar sign from the 'Price' column
df['Price'] = df['Price'].str.replace('$', '')

# Convert the 'Price' column to float
df['Price'] = df['Price'].astype(float)

# Remove rows with missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Separate the input features (X) and the target variable (y)
X = df[['Reviews', 'Size', 'Installs', 'Price']]
y = df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the target variable for the test features
y_pred = model.predict(X_test)

# Calculate the mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict the rating for a new app
new_app_features = np.array([[1000, 25, 100000, 2]])  # Replace with desired input features
predicted_rating = model.predict(new_app_features)
print("Predicted Rating for the New App:", predicted_rating[0])
