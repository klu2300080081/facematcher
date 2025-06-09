
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

# Define VGGNet class
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load VGG16 model
@st.cache_resource
def load_vgg_model():
    model = VGGNet()
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    st.info("Loaded VGG16 model.")
    return model

# Define GenderClassifier (adjust based on training architecture)
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Adjust fc1 input size based on 224x224 input
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Updated for 224x224
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load gender classifier
@st.cache_resource
def load_gender_model():
    model = GenderClassifier()
    try:
        state_dict = torch.load("gender_classifier.pt", map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=False)  # Allow partial loading
        mismatched_keys = [k for k in state_dict.keys() if k not in model.state_dict()]
        if mismatched_keys:
            st.warning(f"Mismatched keys in gender_classifier.pt: {mismatched_keys}")
        model.eval()
        device = torch.device("cpu")
        model.to(device)
        st.info("Loaded gender classifier.")
    except Exception as e:
        st.error(f"Failed to load gender_classifier.pt: {e}")
        return None  # Disable gender prediction if loading fails
    return model

# Preprocess image for gender prediction
def preprocess_gender(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Detect face using Haar Cascade
def detect_face(image):
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        return cv2.resize(face, (224, 224))
    return None

# Extract features using VGG16
def extract_features(vgg_model, image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    device = torch.device("cpu")
    image = image.to(device)
    with torch.no_grad():
        feature = vgg_model(image)
    return feature

# Load dataset
@st.cache_data
def prepare_datasets():
    male_data_path = "Maledataset.pkl"
    female_data_path = "Femaledataset.pkl"
    
    if os.path.exists(male_data_path):
        male_df = pd.read_pickle(male_data_path)
        male_df['image_path'] = male_df['image_path'].apply(
            lambda x: os.path.relpath(x, start=os.getcwd()).replace('\\', '/')
        )
    else:
        male_df = pd.DataFrame()
    
    if os.path.exists(female_data_path):
        female_df = pd.read_pickle(female_data_path)
        female_df['image_path'] = female_df['image_path'].apply(
            lambda x: os.path.relpath(x, start=os.getcwd()).replace('\\', '/')
        )
    else:
        female_df = pd.DataFrame()
    
    return male_df, female_df

# Generate face patterns
@st.cache_data
def generate_face_patterns(vgg_model, male_df, female_df):
    male_patterns_path = "Malepatterns.pkl"
    female_patterns_path = "Femalepatterns.pkl"

    male_patterns_df = None
    female_patterns_df = None

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

    if male_patterns_df is None:
        male_patterns = []
        for idx, row in male_df.iterrows():
            img_path = row['image_path']
            rel_img_path = os.path.relpath(img_path, start=os.getcwd()).replace('\\', '/')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    face = detect_face(img)
                    if face is not None:
                        feature = extract_features(vgg_model, face)
                        male_patterns.append({
                            'name': row['name'],
                            'feature': feature,
                            'image_path': rel_img_path
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

# Predict celebrity
@st.cache_data
def predict_celebrity(input_image, vgg_model, gender_model, male_patterns_df, female_patterns_df):
    face = detect_face(input_image)
    if face is None:
        return None, None, "No face detected"

    if gender_model is None:
        return None, None, "Gender model failed to load"

    # Predict gender
    gender_input = preprocess_gender(face)
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

    top_match = max(similarities, key=lambda x: x[2])
    celebrity_name, celebrity_image_path, similarity = top_match

    return celebrity_name, celebrity_image_path, gender

def main():
    st.title("Celebrity Look-Alike Finder")
    
    # Load models and data
    vgg_model = load_vgg_model()
    gender_model = load_gender_model()
    male_df, female_df = prepare_datasets()
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
            
            # Display celebrity image
            if celebrity_image_path and os.path.exists(celebrity_image_path):
                try:
                    st.image(celebrity_image_path, caption=celebrity_name, use_column_width=True)
                except Exception as e:
                    st.error(f"Failed to display celebrity image: {e}")
                    st.write(f"Celebrity: {celebrity_name}")
            else:
                st.write(f"Celebrity: {celebrity_name} (Image not available)")
        else:
            st.error(gender)  # Error message

if __name__ == "__main__":
    main()
