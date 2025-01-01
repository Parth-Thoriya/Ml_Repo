import streamlit as st

import tensorflow as tf
import numpy as np

# Path to the model file in the repository
MODEL_PATH = "annModel.tflite"

# Load TFLite Model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Prepare Input
def prepare_input(input_data, input_details):
    # Adjust input data shape and type as per model requirements
    input_data = np.array(input_data, dtype=input_details['dtype'])
    input_data = input_data.reshape(input_details['shape'])  # Reshape if required
    return input_data

# Run Inference
def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_data = prepare_input(input_data, input_details)
    interpreter.set_tensor(input_details['index'], input_data)

    interpreter.invoke()  # Run inference
    output_data = interpreter.get_tensor(output_details['index'])  # Get predictions
    return output_data

# Streamlit Interface
def main():
    st.title("Combined Cycle Power Plant")
    st.write("Make Predictions using : Artificial Neural Network.")
    
    input_data = st.text_input("Enter input data as a comma-separated list (e.g., 28.66,   77.95, 1009.56,   69.07)")

    if input_data:
        try:
            # Load the local model
            interpreter = load_tflite_model(MODEL_PATH)

            # Parse input data
            input_data = [float(x.strip()) for x in input_data.split(",")]

            # Run prediction
            predictions = run_inference(interpreter, input_data)

            st.success(f"Net hourly electrical energy output by your power plant in MW is : {str(predictions)[2:-2]}")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

