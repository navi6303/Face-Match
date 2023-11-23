import streamlit as st
import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
import cv2
import face_recognition
import numpy as np
import io
import requests


# Function to load image and send it to the API
def match_faces_with_api(image_file):
    # Prepare the image for API request
    image_np = np.array(Image.open(image_file))
    image_bytes = Image.fromarray(image_np).tobytes()

    # Make a POST request to the API
    api_url = "http://localhost:5000/api/match"  # Update with your API URL
    files = {"file": ("image.jpg", image_bytes)}
    response = requests.post(api_url, files=files)

    # Handle API response
    if response.status_code == 200:
        result = response.json()
        return result.get("most_matched_image"), result.get("accuracy")
    else:
        st.error("Error calling the API")
        return None, None

@st.cache_data
def load_image(image_file):
    try:
        if isinstance(image_file, str):  # If image_file is a file path
            img = Image.open(image_file)
            img_rgb = img.convert("RGB")  # Convert to RGB
            img_np = np.array(img_rgb)  # Convert the image to a NumPy array
        else:  # If image_file is an UploadedFile
            img = Image.open(io.BytesIO(image_file.read()))
            img_rgb = img.convert("RGB")  # Convert to RGB
            img_np = np.array(img_rgb)  # Convert the image to a NumPy array

        return img_np
    except (UnidentifiedImageError, AttributeError) as e:
        st.warning(f"Error loading image: {e}")
        return np.array([])  # Return an empty NumPy array in case of an error

def save_uploaded_file(uploadedfile):
    with open(os.path.join("Dataset", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved Image") 

# Function to find face encodings
def find_face_encodings(image_np):
    if len(image_np.shape) == 2:  # If the image is grayscale, convert it to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    face_enc = face_recognition.face_encodings(image_np)

    if len(face_enc) > 0:
        return face_enc[0]
    else:
        return None  # Return None if no face encodings are found

# Function to find the most matched image and accuracy
def find_most_matched_image(input_image_encoding, dataset_folder):
    most_matched_image = None
    highest_accuracy = 0

    for image_file in os.listdir(dataset_folder):
        image_path = os.path.join(dataset_folder, image_file)
        dataset_image_encoding = find_face_encodings(load_image(image_path))

        # Skip if there are no face encodings for the dataset image
        if dataset_image_encoding is None:
            continue

        is_same = face_recognition.compare_faces([input_image_encoding], dataset_image_encoding)[0]

        if is_same:
            distance = face_recognition.face_distance([input_image_encoding], dataset_image_encoding)
            accuracy = 100 - round(distance[0] * 100)

            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                most_matched_image = image_path

    return most_matched_image, highest_accuracy


# Function to display the result
def display_result(input_image_np, most_matched_image, accuracy):
    st.subheader("Face Matching Result")
    st.image(input_image_np, caption="Input Image", use_column_width=True)
    
    if most_matched_image:
        st.image(Image.open(most_matched_image), caption=f"Most Matched Image (Accuracy: {accuracy}%)", use_column_width=True)
        st.success("The images are a match!")
    else:
        st.warning("No matching image found in the dataset.")

def main():
    st.title("Face Matching")

    menu = ["Home", "Dataset", "Face Match"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Upload Images")
        image_file = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if image_file is not None:
            file_Details = {"FileName": image_file.name, "FileType": image_file.type}
            st.write(file_Details)
            img = load_image(image_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            save_uploaded_file(image_file)

    elif choice == "Dataset":
        st.subheader("Dataset")

        image_folder = "Dataset"
        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

        if not image_files:
            st.info("No images in the dataset folder.")
        else:
            st.subheader("List of Images in Dataset")
            for i, image_file in enumerate(image_files, start=1):
                st.write(f"{i}. {image_file}")

            selected_image_index = st.selectbox("Select an image to display:", range(1, len(image_files) + 1)) - 1
            selected_image = image_files[selected_image_index]

            img_path = os.path.join(image_folder, selected_image)
            img = Image.open(img_path)
            st.image(img, caption=f"Selected Image: {selected_image}", use_column_width=True)
    
    elif choice == "Face Match":
        input_image_file = st.file_uploader("Upload an image for face matching", type=['png', 'jpeg', 'jpg'])

        if input_image_file is not None:
            input_image_np = np.array(Image.open(input_image_file))  # Load the image as a NumPy array

        if len(input_image_np) > 0:  # Check if the image is not an empty NumPy array
            # Call the API for face matching
            most_matched_image, accuracy = match_faces_with_api(input_image_file)

            # Display the result
            st.subheader("Face Matching Result")
            st.image(input_image_np, caption="Input Image", use_column_width=True)

            if most_matched_image:
                st.image(Image.open(most_matched_image), caption=f"Most Matched Image (Accuracy: {accuracy}%)", use_column_width=True)
                st.success("The images are a match!")
            else:
                st.warning("No matching image found in the dataset.")


if __name__ == '__main__':
    main()
