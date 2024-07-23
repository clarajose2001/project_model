import streamlit as st
import pandas as pd
import joblib

# Save the model again with joblib (if needed)
def save_model(model, file_path):
    try:
        joblib.dump(model, file_path)
        st.write("Model saved successfully.")
    except Exception as e:
        st.error(f"Error saving model: {e}")

# Load the trained model
def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model('best_house_price_model.pkl')

# Define the required columns globally
required_columns = [
    "number of bedrooms", "number of bathrooms", "living area", 
    "lot area", "number of floors", "number of views", 
    "Area of the house(excluding basement)", "Area of the basement", 
    "Built Year", "Renovation Year", "living_area_renov", 
    "Number of schools nearby", "Distance from the airport"
]

# Define the Streamlit app
def main():
    st.title("House Price Prediction")

    # Allow users to upload a CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        input_data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded Data")
        st.write(input_data)

        # Make predictions on the uploaded data
        if st.button("Predict from CSV"):
            try:
                # Ensure that the input data has the required columns
                missing_columns = [col for col in required_columns if col not in input_data.columns]
                if missing_columns:
                    st.error(f"The CSV file must contain the following columns: {', '.join(missing_columns)}")
                else:
                    # Ensure column order matches model training
                    input_data = input_data[required_columns]
                    
                    # Make predictions
                    predictions = model.predict(input_data)
                    input_data['Predicted Price'] = predictions
                    st.write("Prediction Results")
                    st.write(input_data)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    st.write("Or, enter data manually:")

    # Define input fields for manual entry
    num_bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3)
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=0, value=2)
    living_area = st.number_input("Living Area (sq ft)", min_value=0, value=1500)
    lot_area = st.number_input("Lot Area (sq ft)", min_value=0, value=5000)
    num_floors = st.number_input("Number of Floors", min_value=1, value=1)
    num_views = st.number_input("Number of Views", min_value=0, value=0)
    area_excluding_basement = st.number_input("Area of the house (excluding basement)", min_value=0, value=1500)
    area_basement = st.number_input("Area of the basement (sq ft)", min_value=0, value=0)
    built_year = st.number_input("Built Year", min_value=1800, max_value=2023, value=2000)
    renovation_year = st.number_input("Renovation Year", min_value=0, max_value=2023, value=0)
    living_area_renov = st.number_input("Living Area Renovation (sq ft)", min_value=0, value=1500)
    num_schools_nearby = st.number_input("Number of Schools Nearby", min_value=0, value=1)
    distance_airport = st.number_input("Distance from the Airport (km)", min_value=0, value=10)

    # Create a prediction button for manual input
    if st.button("Predict Manually"):
        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            "number of bedrooms": [num_bedrooms],
            "number of bathrooms": [num_bathrooms],
            "living area": [living_area],
            "lot area": [lot_area],
            "number of floors": [num_floors],
            "number of views": [num_views],
            "Area of the house(excluding basement)": [area_excluding_basement],
            "Area of the basement": [area_basement],
            "Built Year": [built_year],
            "Renovation Year": [renovation_year],
            "living_area_renov": [living_area_renov],
            "Number of schools nearby": [num_schools_nearby],
            "Distance from the airport": [distance_airport]
        })

        # Make the prediction
        try:
            # Ensure column order matches model training
            input_data = input_data[required_columns]
            
            prediction = model.predict(input_data)
            # Display the prediction result
            st.write(f"Predicted House Price: ₹{prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during manual prediction: {e}")

if __name__ == "__main__":
    if model:
        main()
    else:
        st.error("Model could not be loaded.")
