from roboflow import Roboflow
from dotenv import load_dotenv
import os
import cv2

# Load the .env file
load_dotenv()

# Get the API key from the environment variables
api_key = os.getenv("ROBOFLOW_API_KEY")

# Initialize Roboflow with the API key
rf = Roboflow(api_key=api_key)

# Load the project and model (model endpoint: "chess-piece-detection-5ipnt/3")
project = rf.workspace().project("chess-piece-detection-5ipnt")
model = project.version(3).model

# Define the image file for inference
image_file = "chess.jpg"

# Run inference with confidence and overlap thresholds
result = model.predict(image_file, confidence=40, overlap=30)

# Print the result in JSON format
print(result.json()['predictions'])

# Optionally, save the prediction image
#result.save("prediction.jpg")


