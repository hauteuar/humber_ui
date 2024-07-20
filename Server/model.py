import joblib
import numpy as np
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

# Define the path to the model file
model_path = 'skin_types_image_detection'  # Path to the saved model directory

# Load the trained model
model = ViTForImageClassification.from_pretrained(model_path)

# Define the processor used during training
processor = ViTImageProcessor.from_pretrained(model_path)

# Define the function for making predictions
def predict(image):
    # Preprocess the image for the model
    image = Image.open(image).convert("RGB")  # Open and convert the image to RGB
    inputs = processor(images=image, return_tensors="pt")  # Process the image with the ViT processor
    
    # Perform the prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class label
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    
    # Convert index to label
    id2label = model.config.id2label
    predicted_label = id2label[predicted_class_idx]
    
    return predicted_label



















# import joblib
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier  # what model should come here?

# # Load your pre-trained model (if saved as a file)
# model = joblib.load('skin_types_image_detection_vit.py') # is this correct?

# # Or define your model here if it's not saved as a file
# # model = RandomForestClassifier()
# # model.fit(np.random.rand(10, 4), np.random.randint(0, 2, size=10))  # Example training

# def predict(image):
#     # Process the image and make a prediction (write it)
#     # This is a placeholder, replace it with your actual image processing and prediction code (write it)
#     image_array = np.array(image).reshape(1, -1)  # Example reshaping, replace with your actual processing (write it)
#     prediction = model.predict(image_array)
#     return prediction.tolist()
