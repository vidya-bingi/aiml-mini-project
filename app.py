# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert date to datetime
    df_cleaned = df.drop(columns=['Date', 'Adj Close'])  # Drop unnecessary columns
    return df_cleaned

# App title
st.title("Bitcoin Price Prediction using Linear Regression")

# File upload section
uploaded_file = st.file_uploader("Upload your Bitcoin dataset (CSV format)", type="csv")
if uploaded_file:
    # Preprocess the data
    df_cleaned = load_data(uploaded_file)
    
    # Display dataset
    st.write("### Dataset Preview")
    st.dataframe(df_cleaned.head())

    # Define features (Open, High, Low, Volume) and target (Close)
    X = df_cleaned[['Open', 'High', 'Low', 'Volume']]
    y = df_cleaned['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Performance Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display performance metrics
    st.write("### Performance Metrics")
    st.write(f"- Mean Squared Error: {mse:.2f}")
    st.write(f"- Mean Absolute Error: {mae:.2f}")
    st.write(f"- R-squared: {r2:.2f}")

    # Visualization
    st.write("### Actual vs Predicted Close Price")
    plt.figure(figsize=(10, 5))

    # Sort the actual values and reorder predictions accordingly
    sorted_indices = np.argsort(y_test)  # Indices to sort y_test
    y_test_sorted = np.array(y_test)[sorted_indices]
    y_pred_sorted = np.array(y_pred)[sorted_indices]

    # Scatter plot for actual values (blue points)
    plt.scatter(y_test_sorted, y_test_sorted, color='blue', alpha=0.5, label="Actual Values")

    # Single straight line for predicted values (red line)
    plt.plot(y_test_sorted, y_pred_sorted, color='red', label="Predicted Line")

    plt.xlabel("Actual Close Price")
    plt.ylabel("Predicted Close Price")
    plt.title("Actual vs Predicted Close Price")
    plt.legend()
    st.pyplot(plt)

    # Interactive prediction
    st.write("### Make a Prediction")
    open_price = st.number_input("Enter the Open Price", min_value=0.0)
    high_price = st.number_input("Enter the High Price", min_value=0.0)
    low_price = st.number_input("Enter the Low Price", min_value=0.0)
    volume = st.number_input("Enter the Volume", min_value=0.0)

    if st.button("Predict Close Price"):
        prediction = model.predict([[open_price, high_price, low_price, volume]])
        st.success(f"Predicted Close Price: {prediction[0]:.2f}")
