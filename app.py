import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load pre-trained model and scaler
try:
    model = joblib.load('GB.joblib')  # Ensure the model file is in the same directory
    scaler = joblib.load('scaler.joblib')  # Ensure the scaler file is in the same directory
except FileNotFoundError:
    st.error("Model or scaler file not found. Ensure 'GB.joblib' and 'scaler.joblib' are present in the app directory.")
    st.stop()

st.title("Renewable Energy Consumption Prediction")
st.write("Adjust the sliders to change the input features and see the impact on predicted energy consumption.")

# Interactive sliders for user input
installed_capacity = st.slider("Installed Capacity (MW)", 0.0, 500.0, 50.0, step=10.0)
energy_production = st.slider("Energy Production (MWh)", 0.0, 1000.0, 100.0, step=10.0)
energy_storage_capacity = st.slider("Energy Storage Capacity (MWh)", 0.0, 500.0, 30.0, step=10.0)
storage_efficiency = st.slider("Storage Efficiency (%)", 0.0, 100.0, 80.0, step=1.0)
grid_integration_level = st.slider("Grid Integration Level", 1, 4, 2, step=1)
investment = st.slider("Initial Investment (USD)", 0.0, 1_000_000.0, 500_000.0, step=50_000.0)
financial_incentives = st.slider("Financial Incentives (USD)", 0.0, 500_000.0, 100_000.0, step=10_000.0)
ghg_reduction = st.slider("GHG Emission Reduction (tCO2e)", 0.0, 5000.0, 1000.0, step=50.0)
air_pollution_reduction = st.slider("Air Pollution Reduction Index", 0.0, 100.0, 50.0, step=1.0)

# Create a DataFrame for the user inputs
user_input = pd.DataFrame(
    [[installed_capacity, energy_production, energy_storage_capacity, storage_efficiency,
      grid_integration_level, investment, financial_incentives, ghg_reduction, air_pollution_reduction]],
    columns=['Installed_Capacity_MW', 'Energy_Production_MWh', 'Energy_Storage_Capacity_MWh',
             'Storage_Efficiency_Percentage', 'Grid_Integration_Level', 'Initial_Investment_USD',
             'Financial_Incentives_USD', 'GHG_Emission_Reduction_tCO2e', 'Air_Pollution_Reduction_Index']
)

# Scale the input data
user_input_scaled = scaler.transform(user_input)

# Make predictions
prediction = model.predict(user_input_scaled)

# Display the prediction
st.write(f"Predicted Energy Consumption (MWh): {np.expm1(prediction[0]):.2f}")

# Visualization of the prediction
# Create a range of adjusted predictions for visualization
scaling_factors = np.linspace(0.8, 1.2, 20)  # Factors to adjust a single feature
sample_inputs = []

# Adjust one feature (e.g., Installed Capacity) and keep others constant
for factor in scaling_factors:
    adjusted_input = user_input.copy()
    adjusted_input.iloc[0, 0] *= factor  # Adjust 'Installed Capacity (MW)' (column index 0)
    sample_inputs.append(adjusted_input)

# Concatenate all adjusted inputs into a single DataFrame
sample_inputs_df = pd.concat(sample_inputs, ignore_index=True)

# Scale the adjusted input data
scaled_sample_inputs = scaler.transform(sample_inputs_df)

# Predict for each adjusted input
predictions = model.predict(scaled_sample_inputs)

# Convert predictions back from log scale
predictions = np.expm1(predictions)

# Plot the predictions
fig, ax = plt.subplots()
ax.plot(scaling_factors, predictions, label='Predicted Energy Consumption', color='blue')
ax.set_xlabel("Scaling Factor for Installed Capacity (MW)")
ax.set_ylabel("Predicted Energy Consumption (MWh)")
ax.set_title("Impact of Installed Capacity on Predicted Energy Consumption")
ax.legend()

st.pyplot(fig)

# Select two features to analyze
feature_1 = st.selectbox("Select Feature 1", user_input.columns, index=0)  # Default: first feature
feature_2 = st.selectbox("Select Feature 2", user_input.columns, index=1)  # Default: second feature

# Generate a range of values for the two selected features
feature_1_values = np.linspace(user_input[feature_1].iloc[0] * 0.8, user_input[feature_1].iloc[0] * 1.2, 20)
feature_2_values = np.linspace(user_input[feature_2].iloc[0] * 0.8, user_input[feature_2].iloc[0] * 1.2, 20)

# Create a grid of values
feature_1_grid, feature_2_grid = np.meshgrid(feature_1_values, feature_2_values)

# Flatten the grids for prediction
feature_1_flat = feature_1_grid.ravel()
feature_2_flat = feature_2_grid.ravel()

# Create adjusted inputs for prediction
adjusted_inputs = user_input.copy()
adjusted_inputs = pd.DataFrame(
    np.tile(adjusted_inputs.iloc[0].values, (len(feature_1_flat), 1)),
    columns=user_input.columns
)

# Update the two selected features
adjusted_inputs[feature_1] = feature_1_flat
adjusted_inputs[feature_2] = feature_2_flat

# Scale the adjusted inputs
scaled_inputs = scaler.transform(adjusted_inputs)

# Predict for the adjusted inputs
predictions = model.predict(scaled_inputs)
predictions = np.expm1(predictions)  # Convert back from log scale

# Reshape predictions to match the grid
predictions_grid = predictions.reshape(feature_1_grid.shape)

# Create the 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(feature_1_grid, feature_2_grid, predictions_grid, cmap='viridis', alpha=0.8)

# Add labels and title
ax.set_xlabel(f"{feature_1}")
ax.set_ylabel(f"{feature_2}")
ax.set_zlabel("Predicted Energy Consumption (MWh)")
ax.set_title(f"Interaction between {feature_1} and {feature_2}")

# Display the plot in Streamlit
st.pyplot(fig)
