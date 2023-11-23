from flask import Flask, request, jsonify
import os
from app import load_image, find_face_encodings, find_most_matched_image

app = Flask(__name__)

@app.route('/api/match', methods=['POST'])
def match_faces():
    try:
        # Get the image file from the request
        uploaded_file = request.files['file']
        
        # Load the image and find face encodings
        input_image_np = load_image(uploaded_file)
        input_image_encoding = find_face_encodings(input_image_np)

        if input_image_encoding is not None:
            # Find the most matched image in the dataset and its accuracy
            most_matched_image, accuracy = find_most_matched_image(input_image_encoding, "Dataset")

            # Return the result as JSON
            result = {
                "most_matched_image": most_matched_image,
                "accuracy": accuracy
            }
            return jsonify(result)

        else:
            return jsonify({"error": "No face encoding found in the input image"}), 400

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)