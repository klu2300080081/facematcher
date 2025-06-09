import os
import sys
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
# No longer importing Tkinter specific elements like CENTER, TOP, etc.

# Suppress warnings from TensorFlow (Streamlit handles its own logging)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set Streamlit page configuration
st.set_page_config(
    page_title="Celebrity Look-Alike Finder",
    page_icon="🤩",
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="auto"
)

# --- GLOBAL VARIABLES / ASSETS ---
# Define your category folders here
CATEGORY_FOLDERS = ['Cricketers', 'English', 'Hindi', 'Kannada', 'Malayalam', 'Tamil', 'Telugu']
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'

# --- 1. Dataset Preparation ---
# Use st.cache_data to cache the result of this function, so it runs only once
@st.cache_data
def prepare_datasets(category_folders):
    """
    Prepares the male and female image datasets with relative paths.
    Checks for existing pickle files to avoid re-processing.
    """
    st.info("Preparing datasets (This might take a moment the very first time)...")
    male_data_path = "Maledataset.pkl"
    female_data_path = "Femaledataset.pkl"

    if os.path.exists(male_data_path) and os.path.exists(female_data_path):
        st.success("Maledataset.pkl and Femaledataset.pkl already exist. Loading them.")
        male_df = pd.read_pickle(male_data_path)
        female_df = pd.read_pickle(female_data_path)
        return male_df, female_df

    male_img = []
    female_img = []
    male_names = []
    female_names = []

    current_root = os.getcwd()

    progress_text = "Processing dataset categories: {progress_status}"
    my_bar = st.progress(0.0, text=progress_text.format(progress_status="Starting..."))

    total_categories = len(category_folders)
    for i, category_dir in enumerate(category_folders):
        category_full_path = os.path.join(current_root, category_dir)
        
        my_bar.progress((i + 1) / total_categories, text=progress_text.format(progress_status=f"Category: {category_dir}"))

        if not os.path.isdir(category_full_path):
            st.warning(f"Warning: Category folder '{category_dir}' not found at '{category_full_path}'. Skipping.")
            continue
        
        gender_subdirs = [d for d in os.listdir(category_full_path) if os.path.isdir(os.path.join(category_full_path, d))]

        for gender_dir in gender_subdirs:
            gender_full_path = os.path.join(category_full_path, gender_dir)
            if not os.path.isdir(gender_full_path):
                continue
            
            celeb_subdirs = [d for d in os.listdir(gender_full_path) if os.path.isdir(os.path.join(gender_full_path, d))]

            for celeb_dir in celeb_subdirs:
                celeb_full_path = os.path.join(gender_full_path, celeb_dir)
                if not os.path.isdir(celeb_full_path):
                    continue
                
                imgs = [img_file for img_file in os.listdir(celeb_full_path) if os.path.isfile(os.path.join(celeb_full_path, img_file))]

                for img_file in imgs:
                    relative_path = os.path.relpath(os.path.join(celeb_full_path, img_file), current_root)
                    if gender_dir.upper() == "MALE":
                        male_img.append(relative_path)
                        male_names.append(celeb_dir)
                    else:
                        female_img.append(relative_path)
                        female_names.append(celeb_dir)
    my_bar.empty() # Clear the progress bar when done
    
    male_df = pd.DataFrame({"Name": male_names, "Path": male_img})
    female_df = pd.DataFrame({"Name": female_names, "Path": female_img})

    male_df.to_pickle(male_data_path)
    female_df.to_pickle(female_data_path)
    st.success("Maledataset.pkl and Femaledataset.pkl created.")
    return male_df, female_df

# --- 2. VGG Model and Feature Extraction ---
# Use st.cache_resource to cache models, which are larger objects
@st.cache_resource
def load_vgg_face_model():
    vgg_model_path = "VGGnet.h5"
    if os.path.exists(vgg_model_path):
        st.success("VGGnet.h5 already exists. Loading pre-saved model.")
        # No custom_objects needed for VGG model, it's just a feature extractor
        return load_model(vgg_model_path)

    st.info("Building VGG Face model (first time setup)...")
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    if not os.path.exists('vgg_face_weights.h5'):
        st.error("Error: vgg_face_weights.h5 not found. Please download it from a reliable source and place it in the application directory.")
        st.stop() # Stop the Streamlit app if critical file is missing
    model.load_weights('vgg_face_weights.h5')
    
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    vgg_face_descriptor.save(vgg_model_path) 
    st.success("VGGnet.h5 saved.")
    return vgg_face_descriptor

# Use st.cache_resource for the gender classifier model as well
@st.cache_resource
def load_gender_classifier_model():
    custom_objects_dict = {
        'mae': tf.keras.metrics.MeanAbsoluteError() # Instantiate the class
    }
    try:
        GCmodel = load_model('genderClassifier.h5', custom_objects=custom_objects_dict)
        st.success("Gender Classifier model loaded.")
        return GCmodel
    except Exception as e:
        st.error(f"Error loading genderClassifier.h5: {e}. Please ensure the file is available.")
        st.stop() # Stop the Streamlit app if critical file is missing

# Load the Haar Cascade for face detection
# This does not need caching as it's small and not a Keras model
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
if face_cascade.empty():
    st.error(f"Error: '{HAARCASCADE_PATH}' not found or could not be loaded. Please ensure it's in the same directory.")
    st.stop()


def get_image_pixels_cv2(img_bytes):
    """
    Converts image bytes (from Streamlit upload) to OpenCV format.
    """
    if img_bytes is None:
        return None
    
    # Convert bytes to numpy array
    np_array = np.frombuffer(img_bytes, np.uint8)
    # Decode image
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

def find_face_representation(img, model):
    if img is None:
        return None
    
    try: 
        detected_face = cv2.resize(img, (224, 224))
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 127.5
        img_pixels -= 1
        
        # Use a more robust prediction call (handle potential errors)
        representation = model.predict(img_pixels, verbose=0)
        if representation.shape[0] > 0:
            return representation[0,:]
        else:
            return None # No valid representation
    except Exception as e:
        st.error(f"Error processing face representation: {e}")
        return None

# Caching this function to avoid re-calculating patterns on every rerun IF inputs are same
@st.cache_data
def generate_face_patterns(vgg_model, male_df, female_df):
    """
    Generates face vectors for the male and female datasets.
    Checks for existing pattern pickle files to avoid re-processing.
    """
    male_patterns_path = "Malepatterns.pkl"
    female_patterns_path = "Femalepatterns.pkl"

    if os.path.exists(male_patterns_path) and os.path.exists(female_patterns_path):
        st.success("Malepatterns.pkl and Femalepatterns.pkl already exist. Loading them for patterns.")
        male_patterns_df = pd.read_pickle(male_patterns_path)
        female_patterns_df = pd.read_pickle(female_patterns_path)
        return male_patterns_df, female_patterns_df

    st.info("Generating face patterns for datasets (This might take a while)...")
    # Apply getImagePixels and findFaceRepresentation, handling None values
    # Filter out None paths first to prevent errors
    male_df_valid_paths = male_df[male_df['Path'].apply(lambda p: os.path.exists(os.path.join(os.getcwd(), p)))].copy()
    female_df_valid_paths = female_df[female_df['Path'].apply(lambda p: os.path.exists(os.path.join(os.getcwd(), p)))].copy()


    st.write("Generating patterns for Male dataset...")
    # Use st.progress for better UX during potentially long operations
    male_bar = st.progress(0.0, text="Processing Male images...")
    male_face_vectors = []
    for idx, row in male_df_valid_paths.iterrows():
        img_path = os.path.join(os.getcwd(), row['Path'])
        img = cv2.imread(img_path)
        if img is not None:
            male_face_vectors.append(find_face_representation(img, vgg_model))
        else:
            male_face_vectors.append(None)
        male_bar.progress((idx + 1) / len(male_df_valid_paths), text=f"Processing Male images: {idx+1}/{len(male_df_valid_paths)}")
    male_bar.empty()
    male_df_valid_paths['face_vector_raw'] = male_face_vectors
    male_df_valid_paths.to_pickle(male_patterns_path)
    st.success("Malepatterns.pkl created.")

    st.write("Generating patterns for Female dataset...")
    female_bar = st.progress(0.0, text="Processing Female images...")
    female_face_vectors = []
    for idx, row in female_df_valid_paths.iterrows():
        img_path = os.path.join(os.getcwd(), row['Path'])
        img = cv2.imread(img_path)
        if img is not None:
            female_face_vectors.append(find_face_representation(img, vgg_model))
        else:
            female_face_vectors.append(None)
        female_bar.progress((idx + 1) / len(female_df_valid_paths), text=f"Processing Female images: {idx+1}/{len(female_df_valid_paths)}")
    female_bar.empty()
    female_df_valid_paths['face_vector_raw'] = female_face_vectors
    female_df_valid_paths.to_pickle(female_patterns_path)
    st.success("Femalepatterns.pkl created.")

    return male_df_valid_paths, female_df_valid_paths


def find_cosine_similarity(source_representation, test_representation):
    try:
        source_representation = np.asarray(source_representation)
        test_representation = np.asarray(test_representation)

        a = np.dot(source_representation, test_representation)
        b = np.sum(np.square(source_representation))
        c = np.sum(np.square(test_representation))
        
        if b == 0 or c == 0:
            return 10.0 # Large value for no similarity (or division by zero)
        
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    except Exception as e:
        st.error(f"Error in cosine similarity calculation: {e}")
        return 10.0 
    
def gender_classifier(img_array, gc_model):
    if img_array is None or img_array.size == 0:
        return 0 # Default to female or handle as error

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    resized = resized.reshape(1, 128, 128, 1)
    resized = resized.astype("float32") / 255.0
    
    pred = gc_model.predict(resized, verbose=0) 
    return round(pred[0][0][0]) # Assuming 1 for male, 0 for female


def predict_celebrity(captured_image_cv2, vgg_model, gc_model, male_patterns_df, female_patterns_df):
    if captured_image_cv2 is None:
        return None, "No Image Provided"

    gray_image = cv2.cvtColor(captured_image_cv2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected in the captured image.")
        return None, "No Face Detected"

    # Assuming only one face for simplicity, take the first one
    x, y, w, h = faces[0]
    detected_face = captured_image_cv2[y:y+h, x:x+w]

    # Predict gender
    gender = gender_classifier(detected_face, gc_model)

    # Resize and preprocess for VGG model
    resized_for_vgg = cv2.resize(detected_face, (224, 224))
    img_pixels = image.img_to_array(resized_for_vgg)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 127.5
    img_pixels -= 1
    captured_representation = vgg_model.predict(img_pixels, verbose=0)[0, :] 

    df_patterns = None
    if gender == 1: 
        st.info("Predicted Gender: Male")
        df_patterns = male_patterns_df
    else:
        st.info("Predicted Gender: Female")
        df_patterns = female_patterns_df

    if df_patterns is None or df_patterns.empty:
        return None, "No patterns loaded for this gender."

    # Filter out rows where 'face_vector_raw' is None or empty array
    df_patterns_filtered = df_patterns.dropna(subset=['face_vector_raw'])
    df_patterns_filtered = df_patterns_filtered[df_patterns_filtered['face_vector_raw'].apply(
        lambda x: isinstance(x, np.ndarray) and x.size > 0
    )]

    if df_patterns_filtered.empty:
        st.warning("No valid face vectors found in the filtered patterns dataframe for comparison.")
        return None, "No valid patterns for comparison."

    df_patterns_filtered.loc[:, 'similarity'] = df_patterns_filtered['face_vector_raw'].apply(
        lambda x: find_cosine_similarity(x, captured_representation)
    )
    
    min_similarity_row = df_patterns_filtered[df_patterns_filtered['similarity'] < 0.6].nsmallest(1, 'similarity')

    if not min_similarity_row.empty:
        instance = min_similarity_row.iloc[0]
        full_path = instance['Path']
        celeb_name = instance['Name']
        st.success(f"Closest match: **{celeb_name}** with similarity distance: {instance['similarity']:.4f}")
        return full_path, celeb_name
    else:
        st.info("No strong celebrity match found within threshold.")
        return None, "No Match Found"


# --- Streamlit UI Layout ---
st.title("Celebrity Look-Alike Finder 🤩")

st.markdown("""
Upload an image or capture from your webcam to find your celebrity look-alike!
""")

# Initialize session state variables if they don't exist
if 'predicted_celeb_path' not in st.session_state:
    st.session_state.predicted_celeb_path = None
if 'predicted_celeb_name' not in st.session_state:
    st.session_state.predicted_celeb_name = "N/A"
if 'captured_image_bytes' not in st.session_state:
    st.session_state.captured_image_bytes = None


# Load models and data once
male_df, female_df = prepare_datasets(CATEGORY_FOLDERS)
vgg_model = load_vgg_face_model()
gc_model = load_gender_classifier_model()
male_patterns_df, female_patterns_df = generate_face_patterns(vgg_model, male_df, female_df)


# Tabs for Uploading Image vs. Camera Input
tab1, tab2 = st.tabs(["Upload Image", "Capture from Webcam"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.captured_image_bytes = uploaded_file.read()
        st.image(st.session_state.captured_image_bytes, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Find Look-Alike from Uploaded Image"):
            with st.spinner("Analyzing image..."):
                captured_image_cv2 = get_image_pixels_cv2(st.session_state.captured_image_bytes)
                if captured_image_cv2 is not None:
                    pred_path, pred_name = predict_celebrity(captured_image_cv2, vgg_model, gc_model, male_patterns_df, female_patterns_df)
                    st.session_state.predicted_celeb_path = pred_path
                    st.session_state.predicted_celeb_name = pred_name
                else:
                    st.error("Failed to process uploaded image.")
                    st.session_state.predicted_celeb_path = None
                    st.session_state.predicted_celeb_name = "Error processing image"

with tab2:
    st.subheader("Capture from Webcam")
    camera_image_bytes = st.camera_input("Take a picture")

    if camera_image_bytes is not None:
        st.session_state.captured_image_bytes = camera_image_bytes.read()
        st.image(st.session_state.captured_image_bytes, caption="Captured Image", use_column_width=True)

        if st.button("Find Look-Alike from Captured Image"):
            with st.spinner("Analyzing image..."):
                captured_image_cv2 = get_image_pixels_cv2(st.session_state.captured_image_bytes)
                if captured_image_cv2 is not None:
                    pred_path, pred_name = predict_celebrity(captured_image_cv2, vgg_model, gc_model, male_patterns_df, female_patterns_df)
                    st.session_state.predicted_celeb_path = pred_path
                    st.session_state.predicted_celeb_name = pred_name
                else:
                    st.error("Failed to process captured image.")
                    st.session_state.predicted_celeb_path = None
                    st.session_state.predicted_celeb_name = "Error processing image"


st.markdown("---")
st.header("Your Celebrity Look-Alike Prediction")

# Display the prediction result
if st.session_state.predicted_celeb_path:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Input Image")
        if st.session_state.captured_image_bytes:
            st.image(st.session_state.captured_image_bytes, caption="Your Image", use_column_width=True)
        else:
            st.write("No image captured yet.")
    with col2:
        st.subheader("Predicted Celebrity Match")
        try:
            # Load the predicted celebrity image for display
            predicted_img_display_path = os.path.join(os.getcwd(), st.session_state.predicted_celeb_path)
            if os.path.exists(predicted_img_display_path):
                st.image(predicted_img_display_path, caption=st.session_state.predicted_celeb_name, use_column_width=True)
            else:
                st.warning(f"Predicted celebrity image not found at: {predicted_img_display_path}")
                st.info("Check your dataset paths and file existence.")
            st.write(f"**Match:** {st.session_state.predicted_celeb_name}")
        except Exception as e:
            st.error(f"Error displaying predicted celebrity image: {e}")
            st.write(f"**Match:** {st.session_state.predicted_celeb_name} (Image could not be displayed)")
else:
    st.write("Click 'Find Look-Alike' after uploading/capturing an image to see the result.")
    if st.session_state.predicted_celeb_name != "N/A":
        st.write(f"**Status:** {st.session_state.predicted_celeb_name}")