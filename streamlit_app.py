
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import io
import pickle

# ... other imports and functions (e.g., VGGNet, load_gender_model, detect_face, extract_features) ...

@st.cache_data
def generate_face_patterns(vgg_model, male_df, female_df):
    male_patterns_path = "Malepatterns.pkl"
    female_patterns_path = "Femalepatterns.pkl"

    male_patterns_df = None
    female_patterns_df = None

    # Try loading cached patterns
    if os.path.exists(male_patterns_path):
        try:
            male_patterns_df = pd.read_pickle(male_patterns_path)
            st.info("Loaded male patterns from cache.")
        except (ImportError, AttributeError, pickle.UnpicklingError) as e:
            st.warning(f"Failed to load Malepatterns.pkl: {e}. Regenerating patterns.")
            male_patterns_df = None

    if os.path.exists(female_patterns_path):
        try:
            female_patterns_df = pd.read_pickle(female_patterns_path)
            st.info("Loaded female patterns from cache.")
        except (ImportError, AttributeError, pickle.UnpicklingError) as e:
            st.warning(f"Failed to load Femalepatterns.pkl: {e}. Regenerating patterns.")
            female_patterns_df = None

    # Regenerate patterns if not loaded
    if male_patterns_df is None:
        male_patterns = []
        for idx, row in male_df.iterrows():
            img_path = row['image_path']  # e.g., absolute or dataset path
            # Convert to relative path from repo root
            rel_img_path = os.path.relpath(img_path, start=os.getcwd()).replace('\\', '/')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    face = detect_face(img)  # Your face detection function
                    if face is not None:
                        feature = extract_features(vgg_model, face)  # Your feature extraction
                        male_patterns.append({
                            'name': row['name'],
                            'feature': feature,
                            'image_path': rel_img_path  # Store relative path
                        })
        male_patterns_df = pd.DataFrame(male_patterns)
        male_patterns_df.to_pickle(male_patterns_path)
        st.success("Generated and saved male patterns.")

    if female_patterns_df is None:
        female_patterns = []
        for idx, row in female_df.iterrows():
            img_path = row['image_path']
            rel_img_path = os.path.relpath(img_path, start=os.getcwd()).replace('\\', '/')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    face = detect_face(img)
                    if face is not None:
                        feature = extract_features(vgg_model, face)
                        female_patterns.append({
                            'name': row['name'],
                            'feature': feature,
                            'image_path': rel_img_path
                        })
        female_patterns_df = pd.DataFrame(female_patterns)
        female_patterns_df.to_pickle(female_patterns_path)
        st.success("Generated and saved female patterns.")

    return male_patterns_df, female_patterns_df

@st.cache_data
def predict_celebrity(input_image, vgg_model, gender_model, male_patterns_df, female_patterns_df):
    # Preprocess input image
    face = detect_face(input_image)
    if face is None:
        return None, None, "No face detected"

    # Predict gender
    gender_input = preprocess_gender(face)  # Your preprocessing
    with torch.no_grad():
        gender_output = gender_model(gender_input)
        gender_prob = torch.softmax(gender_output, dim=1)
        gender_idx = torch.argmax(gender_prob).item()
        gender = "Male" if gender_idx == 0 else "Female"

    # Extract features
    input_feature = extract_features(vgg_model, face)

    # Compare with patterns
    patterns_df = male_patterns_df if gender == "Male" else female_patterns_df
    similarities = []
    for idx, row in patterns_df.iterrows():
        pattern_feature = row['feature']
        similarity = torch.cosine_similarity(input_feature, pattern_feature, dim=1).item()
        similarities.append((row['name'], row['image_path'], similarity))

    if not similarities:
        return None, None, f"No {gender} celebrities found"

    # Get top match
    top_match = max(similarities, key=lambda x: x[2])
    celebrity_name, celebrity_image_path, similarity = top_match

    return celebrity_name, celebrity_image_path, gender

def main():
    st.title("Celebrity Look-Alike Finder")
    
    # Load models and data
    vgg_model = load_vgg_model()  # Your VGG16 model
    gender_model = load_gender_model()  # Your gender classifier
    male_df, female_df = prepare_datasets()  # Your dataset loader
    male_patterns_df, female_patterns_df = generate_face_patterns(vgg_model, male_df, female_df)

    # Image input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    use_camera = st.checkbox("Use camera")
    
    input_image = None
    if use_camera:
        camera_image = st.camera_input("Take a photo")
        if camera_image:
            input_image = Image.open(camera_image).convert('RGB')
            input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    elif uploaded_file:
        input_image = Image.open(uploaded_file).convert('RGB')
        input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    if input_image is not None and st.button("Find Look-Alike"):
        celebrity_name, celebrity_image_path, gender = predict_celebrity(
            input_image, vgg_model, gender_model, male_patterns_df, female_patterns_df
        )
        
        if celebrity_name:
            st.write(f"Predicted Gender: {gender}")
            st.write(f"Closest Celebrity: {celebrity_name}")
            
            # Display input image
            st.image(input_image, caption="Input Image", use_column_width=True)
            
            # Display celebrity image with error handling
            if celebrity_image_path and os.path.exists(celebrity_image_path):
                try:
                    st.image(celebrity_image_path, caption=celebrity_name, use_column_width=True)
                except Exception as e:
                    st.error(f"Failed to display celebrity image: {e}")
                    st.write(f"Celebrity: {celebrity_name}")
            else:
                st.write(f"Celebrity: {celebrity_name} (Image not available)")
        else:
            st.error(gender)  # Error message like "No face detected"

if __name__ == "__main__":
    main()
