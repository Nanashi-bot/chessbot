from inference import get_model
import supervision as sv
import cv2
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the ROBOFLOW_API_KEY environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")
print(api_key)  # Verify if the key is loaded correctly



# define the image url to use for inference
image_file = "chess.jpeg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_model(model_id="chess-piece-detection-5ipnt/3")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0]

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)


