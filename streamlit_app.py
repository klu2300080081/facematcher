import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import io
from torchvision import transforms
import streamlit.watcher.local_sources_watcher as watcher

# Patch Streamlit file watcher to avoid torch.classes error
original_get_module_paths = watcher.get_module_paths
def patched_get_module_paths(module):
    if module.__name__.startswith("torch.classes"):
        return []
    return original_get_module_paths(module)
watcher.get_module_paths = patched_get_module_paths

# Set Streamlit page configuration
st.set_page_config(
    page_title="Celebrity Look-Alike Finder",
    page_icon="🤩",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- GLOBAL VARIABLES ---
CATEGORY_FOLDERS = ['Cricketers', 'English', 'Hindi', 'Kannada', 'Malayalam', 'Tamil', 'Telugu']
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'

# --- VGG Face Model Definition ---
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 2622, kernel_size=1),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Gender Classifier Model Definition ---
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- Helper Functions ---
@st.cache_data
def prepare_datasets(category_folders):
    male_data_path = "Maledataset.pkl"
    female_data_path = "Femaledataset.pkl"

    if os.path.exists(male_data_path) and os.path.exists(female_data_path):
        st.success("Loading existing dataset pickle files.")
        return pd.read_pickle(male_data_path), pd.read_pickle(female_data_path)

    male_img, female_img = [], []
    male_names, female_names = [], []
    current_root = os.getcwd()

    progress_text = "Processing dataset: {progress_status}"
    my_bar = st.progress(0.0, text=progress_text.format(progress_status="Starting..."))

    for i, category_dir in enumerate(category_folders):
        category_path = os.path.join(current_root, category_dir)
        my_bar.progress((i + 1) / len(category_folders), text=progress_text.format(progress_status=category_dir))
        
        if not os.path.isdir(category_path):
            st.warning(f"Category folder '{category_dir}' not found. Skipping.")
            continue
        
        for gender_dir in os.listdir(category_path):
            gender_path = os.path.join(category_path, gender_dir)
            if not os.path.isdir(gender_path):
                continue
                
            for celeb_dir in os.listdir(gender_path):
                celeb_path = os.path.join(gender_path, celeb_dir)
                if not os.path.isdir(celeb_path):
                    continue
                    
                for img_file in os.listdir(celeb_path):
                    if os.path.isfile(os.path.join(celeb_path, img_file)):
                        relative_path = os.path.relpath(os.path.join(celeb_path, img_file), current_root)
                        if gender_dir.upper() == "MALE":
                            male_img.append(relative_path)
                            male_names.append(celeb_dir)
                        else:
                            female_img.append(relative_path)
                            female_names.append(celeb_dir)
    
    my_bar.empty()
    male_df = pd.DataFrame({"Name": male_names, "Path": male_img})
    female_df = pd.DataFrame({"Name": female_names, "Path": female_img})
    male_df.to_pickle(male_data_path)
    female_df.to_pickle(female_data_path)
    st.success("Dataset pickle files created.")
    return male_df, female_df

@st.cache_resource
def load_vgg_model():
    model_path = "vgg_face.pt"
    if os.path.exists(model_path):
        model = VGGNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        st.success("Loaded VGG face model.")
        return model
    st.error("VGG face model (vgg_face.pt) not found. Please ensure it is in the repository.")
    st.stop()

@st.cache_resource
def load_gender_model():
    model_path = "gender_classifier.pt"
    if os.path.exists(model_path):
        model = GenderClassifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        st.success("Loaded gender classifier model.")
        return model
    st.error("Gender classifier model (gender_classifier.pt) not found. Please ensure it is in the repository.")
    st.stop()

@st.cache_resource
def generate_face_patterns(_vgg_model, male_df, female_df):
    male_patterns_path = "Malepatterns.pkl"
    female_patterns_path = "Femalepatterns.pkl"

    if os.path.exists(male_patterns_path) and os.path.exists(female_patterns_path):
        st.success("Loading existing face pattern pickle files.")
        male_patterns_df = pd.read_pickle(male_patterns_path)
        female_patterns_df = pd.read_pickle(female_patterns_path)
        if not (male_patterns_df.empty or female_patterns_df.empty):
            return male_patterns_df, female_patterns_df
        st.warning("Loaded pickle files are empty. Regenerating patterns...")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    male_df_valid = male_df[male_df['Path'].apply(lambda p: os.path.exists(os.path.join(os.getcwd(), p)))].copy()
    female_df_valid = female_df[female_df['Path'].apply(lambda p: os.path.exists(os.path.join(os.getcwd(), p)))].copy()

    if male_df_valid.empty and female_df_valid.empty:
        st.error("No valid image paths found in dataset. Please check dataset folders.")
        st.stop()

    def extract_features(df, name):
        face_vectors = []
        bar = st.progress(0.0, text=f"Processing {name} images...")
        for idx, row in df.iterrows():
            img_path = os.path.join(os.getcwd(), row['Path'])
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    features = _vgg_model(img_tensor).numpy().flatten()
                face_vectors.append(features)
            else:
                face_vectors.append(None)
            bar.progress((idx + 1) / len(df))
        bar.empty()
        df['face_vector_raw'] = face_vectors
        return df

    st.info("Generating face patterns...")
    male_df_valid = extract_features(male_df_valid, "Male")
    female_df_valid = extract_features(female_df_valid, "Female")
    male_df_valid.to_pickle(male_patterns_path)
    female_df_valid.to_pickle(female_patterns_path)
    st.success("Face pattern pickle files created.")
    return male_df_valid, female_df_valid

def get_image_pixels(img_bytes):
    if img_bytes is None:
        return None
    np_array = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def find_face_representation(img, model):
    if img is None:
        return None
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            return model(img_tensor).numpy().flatten()
    except Exception as e:
        st.error(f"Error processing face representation: {e}")
        return None

def find_cosine_similarity(source, test):
    try:
        source = np.asarray(source)
        test = np.asarray(test)
        a = np.dot(source, test)
        b = np.sum(np.square(source))
        c = np.sum(np.square(test))
        if b == 0 or c == 0:
            return 10.0
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    except Exception as e:
        st.error(f"Error in cosine similarity: {e}")
        return 10.0

def gender_classifier(img_array, gc_model):
    if img_array is None or img_array.size == 0:
        return 0
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img_tensor = transform(resized).unsqueeze(0)
    with torch.no_grad():
        pred = gc_model(img_tensor).item()
    return round(pred)

def predict_celebrity(img, vgg_model, gc_model, male_patterns_df, female_patterns_df):
    if img is None:
        return None, "No Image Provided", None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if face_cascade.empty():
        st.error("Haar cascade file not found.")
        st.stop()
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        st.warning("No face detected.")
        return None, "No Face Detected", None
    
    x, y, w, h = faces[0]
    detected_face = img[y:y+h, x:x+w]
    gender = gender_classifier(detected_face, gc_model)
    
    df_patterns = male_patterns_df if gender == 1 else female_patterns_df
    
    if df_patterns.empty:
        st.warning("No patterns available for this gender.")
        return None, "No patterns for this gender.", gender
    
    df_patterns = df_patterns.dropna(subset=['face_vector_raw'])
    df_patterns = df_patterns[df_patterns['face_vector_raw'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)]
    
    if df_patterns.empty:
        st.warning("No valid face vectors for comparison.")
        return None, "No valid patterns for comparison.", gender
    
    representation = find_face_representation(detected_face, vgg_model)
    if representation is None:
        st.warning("Failed to extract face representation.")
        return None, "Error extracting face representation.", gender
    
    df_patterns['similarity'] = df_patterns['face_vector_raw'].apply(
        lambda x: find_cosine_similarity(x, representation)
    )
    
    # Find closest match (increased threshold to 0.9)
    min_similarity_row = df_patterns[df_patterns['similarity'] < 0.9].nsmallest(1, 'similarity')
    
    if not min_similarity_row.empty:
        instance = min_similarity_row.iloc[0]
        img_path = os.path.join(os.getcwd(), instance['Path'])
        if os.path.exists(img_path):
            return instance['Path'], instance['Name'], gender
        else:
            return None, f"Image not found for {instance['Name']}", gender
    else:
        # Return closest match if no threshold met
        if not df_patterns.empty:
            best_match = df_patterns.nsmallest(1, 'similarity').iloc[0]
            img_path = os.path.join(os.getcwd(), best_match['Path'])
            if os.path.exists(img_path):
                return best_match['Path'], best_match['Name'], gender
            else:
                return None, f"Image not found for {best_match['Name']}", gender
        return None, "No Match Found", gender

# --- Streamlit UI ---
st.title("Celebrity Look-Alike Finder 🤩")
st.markdown("Upload an image or use your webcam to find your celebrity look-alike!")

# Initialize session state
if 'predicted_celeb_path' not in st.session_state:
    st.session_state.predicted_celeb_path = None
if 'predicted_celeb_name' not in st.session_state:
    st.session_state.predicted_celeb_name = "N/A"
if 'captured_image_bytes' not in st.session_state:
    st.session_state.captured_image_bytes = None
if 'predicted_gender' not in st.session_state:
    st.session_state.predicted_gender = None

# Load models and data
male_df, female_df = prepare_datasets(CATEGORY_FOLDERS)
vgg_model = load_vgg_model()
gc_model = load_gender_model()
male_patterns_df, female_patterns_df = generate_face_patterns(vgg_model, male_df, female_df)

# Tabs for input
tab1, tab2 = st.tabs(["Upload Image", "Capture Image"])

with tab1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.captured_image_bytes = uploaded_file.read()
        st.image(st.session_state.captured_image_bytes, caption="Uploaded Image", use_column_width=True)
        if st.button("Find Look-Alike (Image)"):
            with st.spinner("Analyzing..."):
                img = get_image_pixels(st.session_state.captured_image_bytes)
                pred_path, pred_name, gender = predict_celebrity(img, vgg_model, gc_model, male_patterns_df, female_patterns_df)
                st.session_state.predicted_celeb_path = pred_path
                st.session_state.predicted_celeb_name = pred_name
                st.session_state.predicted_gender = "Male" if gender == 1 else "Female" if gender == 0 else None
                if st.session_state.predicted_gender:
                    st.info(f"Predicted Gender: {st.session_state.predicted_gender}")

with tab2:
    st.subheader("Capture Image")
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        st.session_state.captured_image_bytes = camera_image.read()
        st.image(st.session_state.captured_image_bytes, caption="Captured Image", use_column_width=True)
        if st.button("Find Look-Alike (Camera)"):
            with st.spinner("Analyzing..."):
                img = get_image_pixels(st.session_state.captured_image_bytes)
                pred_path, pred_name, gender = predict_celebrity(img, vgg_model, gc_model, male_patterns_df, female_patterns_df)
                st.session_state.predicted_celeb_path = pred_path
                st.session_state.predicted_celeb_name = pred_name
                st.session_state.predicted_gender = "Male" if gender == 1 else "Female" if gender == 0 else None
                if st.session_state.predicted_gender:
                    st.info(f"Predicted Gender: {st.session_state.predicted_gender}")

# --- Display Results ---
st.markdown("")
st.header("Your Celebrity Match")
if st.session_state.predicted_gender:
    st.info(f"Gender: {st.session_state.predicted_gender}")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Image")
    if st.session_state.captured_image_bytes:
        try:
            img = Image.open(io.BytesIO(st.session_state.captured_image_bytes))
            st.image(img, caption="Your Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying your image: {e}")
    else:
        st.write("No image uploaded. Please upload or capture an image.")

with col2:
    st.subheader("Celebrity Match")
    if st.session_state.predicted_celeb_path:
        img_path = os.path.join(os.getcwd(), st.session_state.predicted_celeb_path)
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path)
                st.image(image, caption=st.session_state.predicted_celeb_name, use_column_width=True)
                st.write(f"**Name:** {st.session_state.predicted_celeb_name}")
            except Exception as e:
                st.error(f"Error displaying celebrity image: {e}")
                st.write(f"**Name:** {st.session_state.predicted_celeb_name}")
        else:
            st.error(f"Celebrity image not found: {img_path}")
            st.write(f"**Name:** {st.session_state.predicted_celeb_name}")
    else:
        st.write("No match found. Try a different image.")
        if st.session_state.predicted_celeb_name != "N/A":
            st.write(f"**Status:** {st.session_state.predicted_celeb_name}")