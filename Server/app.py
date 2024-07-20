from flask import Flask, request, jsonify
from model import predict  # Import the predict function from model.py

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    # Get the image from the POST request
    file = request.files['image']
    # Call the predict function from model.py
    result = predict(file)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

























# # server/main.py

# from flask import Flask, request, jsonify
# from model import predict  # Import the predict function from the model.py file

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file provided'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Read the image file and process it
#     image = file.read()  # Read the file content
#     # Convert image bytes to a format suitable for your model (this will depend on how your model processes images)
#     # You might need to save the image to a temporary file or convert it into a numpy array

#     prediction = predict(image)  # Call the predict function from model.py
#     return jsonify({'prediction': prediction})

# if __name__ == '__main__':
#     app.run(debug=True)
