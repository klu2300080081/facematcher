import os
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import Tk, Label, Frame, Button, CENTER, TOP, BOTTOM, LEFT, RIGHT # <-- ADD THESE

# Suppress warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.utils import load_img 

sys.stdout.reconfigure(encoding='utf-8')

# ... (rest of your code, which should be the same as the last version I provided,
# but with the fixed import)

# --- 1. Dataset Preparation (from mFdataset.ipynb) ---
def prepare_datasets(category_folders):
    """
    Prepares the male and female image datasets with relative paths.
    Checks for existing pickle files to avoid re-processing.
    
    Args:
        category_folders (list): A list of top-level folders containing gender subfolders.
    """
    print("Preparing datasets...")
    male_data_path = "Maledataset.pkl"
    female_data_path = "Femaledataset.pkl"

    if os.path.exists(male_data_path) and os.path.exists(female_data_path):
        print("Maledataset.pkl and Femaledataset.pkl already exist. Loading them.")
        male_df = pd.read_pickle(male_data_path)
        female_df = pd.read_pickle(female_data_path)
        return male_df, female_df

    male_img = []
    female_img = []
    male_names = []
    female_names = []

    current_root = os.getcwd() # Get the initial working directory

    for category_dir in category_folders: # e.g., "Cricketers", "English"
        category_full_path = os.path.join(current_root, category_dir)
        if not os.path.isdir(category_full_path):
            print(f"Warning: Category folder '{category_dir}' not found at '{category_full_path}'. Skipping.")
            continue
        
        print(f"Processing category: {category_dir}")
        
        gender_subdirs = [d for d in os.listdir(category_full_path) if os.path.isdir(os.path.join(category_full_path, d))]

        for gender_dir in gender_subdirs: # e.g., "Male", "Female"
            gender_full_path = os.path.join(category_full_path, gender_dir)
            if not os.path.isdir(gender_full_path):
                continue
            
            celeb_subdirs = [d for d in os.listdir(gender_full_path) if os.path.isdir(os.path.join(gender_full_path, d))]

            for celeb_dir in celeb_subdirs: # e.g., "Brad Pitt"
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
    
    male_df = pd.DataFrame({"Name": male_names, "Path": male_img})
    female_df = pd.DataFrame({"Name": female_names, "Path": female_img})

    male_df.to_pickle(male_data_path)
    female_df.to_pickle(female_data_path)
    print("Maledataset.pkl and Femaledataset.pkl created.")
    return male_df, female_df

# --- 2. VGG Model and Feature Extraction (from prototype2.ipynb) ---

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def loadVggFaceModel():
    vgg_model_path = "VGGnet.h5"
    if os.path.exists(vgg_model_path):
        print("VGGnet.h5 already exists. Loading pre-saved model.")
        return load_model(vgg_model_path)

    print("Building VGG Face model...")
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
    
    # Load pretrained weights
    if not os.path.exists('vgg_face_weights.h5'):
        print("Error: vgg_face_weights.h5 not found. Please download it from a reliable source.")
        sys.exit(1) # Exit if weights are missing
    model.load_weights('vgg_face_weights.h5')
    
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
    # WARNING:absl:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
    # This warning means the optimizer, loss, and metrics were not saved.
    # If this model is *only* for feature extraction (like VGG Face for embeddings),
    # then compilation is not strictly necessary for its usage.
    # If it was trained and needs to be continued training, you'd compile it here.
    # For now, let's just save it.
    vgg_face_descriptor.save(vgg_model_path) 
    print("VGGnet.h5 saved.")
    return vgg_face_descriptor

def getImagePixels(image_path):
    """Loads an image from a relative path, ensuring it's from the current working directory."""
    full_path = os.path.join(os.getcwd(), image_path)
    try:
        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: Could not read image at {full_path}. Check file existence/corruption.")
        return img
    except Exception as e:
        print(f"Error loading image {full_path}: {e}")
        return None

def findFaceRepresentation(img, model):
    if img is None:
        return None
    detected_face = img
    
    try: 
        detected_face = cv2.resize(detected_face, (224, 224))
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 127.5
        img_pixels -= 1
        
        representation = model.predict(img_pixels, verbose=0)[0,:] 
    except Exception as e:
        print(f"Error processing face representation: {e}")
        representation = None
        
    return representation

def generate_face_patterns(vgg_model, male_df, female_df):
    """
    Generates face vectors for the male and female datasets.
    Checks for existing pattern pickle files to avoid re-processing.
    """
    male_patterns_path = "Malepatterns.pkl"
    female_patterns_path = "Femalepatterns.pkl"

    if os.path.exists(male_patterns_path) and os.path.exists(female_patterns_path):
        print("Malepatterns.pkl and Femalepatterns.pkl already exist. Loading them.")
        male_patterns_df = pd.read_pickle(male_patterns_path)
        female_patterns_df = pd.read_pickle(female_patterns_path)
        return male_patterns_df, female_patterns_df

    print("Generating face patterns for Male dataset...")
    male_df['pixels'] = male_df['Path'].apply(getImagePixels)
    male_df['face_vector_raw'] = male_df['pixels'].apply(lambda x: findFaceRepresentation(x, vgg_model))
    male_df.to_pickle(male_patterns_path)
    print("Malepatterns.pkl created.")

    print("Generating face patterns for Female dataset...")
    female_df['pixels'] = female_df['Path'].apply(getImagePixels)
    female_df['face_vector_raw'] = female_df['pixels'].apply(lambda x: findFaceRepresentation(x, vgg_model))
    female_df.to_pickle(female_patterns_path)
    print("Femalepatterns.pkl created.")

    return male_df, female_df


# --- 3. Final Output and GUI (from final.py) ---
class VideoCaptureApp:
    def __init__(self):
        # Define custom objects dictionary for model loading
        # Based on your error message, it expects the class 'MeanAbsoluteError'
        # The key should be what the model was compiled with (likely 'mae')
        custom_objects_dict = {
            'mae': tf.keras.metrics.MeanAbsoluteError() # Instantiate the class
        }
        
        # Load models
        try:
            self.GCmodel = load_model('genderClassifier.h5', custom_objects=custom_objects_dict)
            self.VGGmodel = load_model("VGGnet.h5")
        except Exception as e:
            print(f"Error loading models: {e}. Please ensure 'genderClassifier.h5' and 'VGGnet.h5' are available.")
            sys.exit(1)

        self.root = Tk()
        self.root.configure(bg="blue")
        
        self.screen_width = self.root.winfo_screenwidth() - 100
        self.screen_height = self.root.winfo_screenheight() - 100
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+50+50")
        
        self.frame_width = 500
        self.frame_height = 600

        self.headText = Label(self.root, text="Celebrity Look-Alike Finder", justify=CENTER, font=("Arial", 24, "bold"), fg="white", bg="blue")
        self.headText.pack(padx=10, pady=10)

        self.constructLeftFrame()
        self.caputure()
        self.constructRightFrame()
        
        self.submit_button = Button(self.root, text="Capture & Predict", command=self.display, justify=CENTER, font=("Arial", 16), bg="green", fg="white", activebackground="darkgreen", activeforeground="white")
        self.submit_button.pack(side=BOTTOM, pady=20)
        
        # Load pattern datasets for prediction
        try:
            self.male_patterns_df = pd.read_pickle("Malepatterns.pkl")
            self.female_patterns_df = pd.read_pickle("Femalepatterns.pkl")
        except FileNotFoundError:
            print("Pattern files not found. Please ensure 'Malepatterns.pkl' and 'Femalepatterns.pkl' are generated.")
            sys.exit(1)


    def constructLeftFrame(self):
        self.leftFrame = Frame(self.root, width=self.frame_width, height=self.frame_height, bg="lightgray", bd=2, relief="groove")
        self.leftFrame.pack(side=LEFT, padx=50, pady=20)
        self.leftFrame.pack_propagate(False)

        self.leftHead = Label(self.leftFrame, text="Live Camera Feed", justify=CENTER, font=("Arial", 18, "bold"), bg="lightgray")
        self.leftHead.pack(side=TOP, pady=5)

    def constructRightFrame(self):
        self.rightFrame = Frame(self.root, width=self.frame_width, height=self.frame_height, bg="lightgray", bd=2, relief="groove")
        self.rightFrame.pack(side=RIGHT, padx=50, pady=20)
        self.rightFrame.pack_propagate(False)

        self.rightHead = Label(self.rightFrame, text="Captured Image & Prediction", justify=CENTER, font=("Arial", 18, "bold"), bg="lightgray")
        self.rightHead.pack(side=TOP, pady=5)
        
        self.captured = None
        self.image_frame = self.rightFrame
        self.imglabel = Label(self.image_frame, bg="lightgray")
        self.imglabel.pack(fill="both", expand=True, padx=10, pady=10)

        self.prediction_text = Label(self.rightFrame, text="Prediction: N/A", font=("Arial", 16), bg="lightgray", fg="black")
        self.prediction_text.pack(side=BOTTOM, pady=5)


    def caputure(self, video_source=0):
        self.video_frame = self.leftFrame
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            # Try other common sources if default fails, or exit
            for i in range(1, 5): # Try up to 4 other cameras
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"Opened camera on source {i}")
                    break
            if not self.cap.isOpened():
                print("No active camera found. Exiting.")
                sys.exit(1)

        self.label = Label(self.video_frame, bg="black")
        self.label.pack(fill="both", expand=True, padx=10, pady=10)
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame.copy()
            # Resize frame to fit the display area, maintaining aspect ratio
            h, w, _ = frame.shape
            ratio = min(self.frame_height / h, self.frame_width / w)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            # Avoid error if new_w or new_h become 0 due to very small ratio or invalid frame
            if new_w == 0 or new_h == 0:
                self.video_frame.after(15, self.update)
                return

            resized_frame = cv2.resize(frame, (new_w, new_h))

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
            self.label.configure(image=self.photo)
            self.label.image = self.photo
        self.video_frame.after(15, self.update)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened(): 
            self.cap.release()

    def display(self):
        if self.last_frame is None:
            self.prediction_text.config(text="No frame captured yet.")
            return

        self.captured = self.last_frame.copy()
        self.predCeleb_path, self.predCeleb_name = self.predictCelebrity()

        if self.predCeleb_path:
            try:
                # Load the predicted celebrity image
                predicted_img_cv2 = cv2.imread(os.path.join(os.getcwd(), self.predCeleb_path))
                if predicted_img_cv2 is None:
                    print(f"Warning: Could not load predicted image from {self.predCeleb_path}")
                    self.prediction_text.config(text=f"Prediction: {self.predCeleb_name} (Image not found)")
                    # Optionally, keep the captured image or clear the right frame
                    img_display = Image.fromarray(cv2.cvtColor(self.captured, cv2.COLOR_BGR2RGB))
                else:
                    # Convert OpenCV image to PIL Image for display
                    img_display = Image.fromarray(cv2.cvtColor(predicted_img_cv2, cv2.COLOR_BGR2RGB))

                # Resize the predicted image for display in the right frame
                # CORRECTED LINE HERE:
                w, h = img_display.size # PIL Image.size returns (width, height)

                ratio = min((self.frame_height - 100) / h, (self.frame_width - 100) / w)
                new_w = int(w * ratio)
                new_h = int(h * ratio)

                if new_w == 0 or new_h == 0:
                    self.prediction_text.config(text=f"Prediction: {self.predCeleb_name} (Error resizing image)")
                    return

                img_display = img_display.resize((new_w, new_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_display)
                self.imglabel.configure(image=photo)
                self.imglabel.image = photo
                self.prediction_text.config(text=f"Prediction: {self.predCeleb_name}")

            except Exception as e:
                print(f"Error displaying predicted image: {e}")
                self.prediction_text.config(text=f"Prediction: {self.predCeleb_name} (Error displaying image)")
        else:
            # If no prediction was made, you can clear the right frame or display a default message
            self.imglabel.config(image=None)
            self.imglabel.image = None
            self.prediction_text.config(text="Prediction: Could not find a match.")


    def findCosineSimilarity(self, source_representation, test_representation):
        try:
            source_representation = np.asarray(source_representation)
            test_representation = np.asarray(test_representation)

            a = np.dot(source_representation, test_representation)
            b = np.sum(np.square(source_representation))
            c = np.sum(np.square(test_representation))
            
            if b == 0 or c == 0:
                return 10.0 
            
            return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
        except Exception as e:
            print(f"Error in cosine similarity calculation: {e}")
            return 10.0 
        
    def genderClassifier(self, img_array):
        if img_array is None or img_array.size == 0:
            return 0 

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        resized = resized.reshape(1, 128, 128, 1)
        resized = resized.astype("float32") / 255.0
        
        pred = self.GCmodel.predict(resized, verbose=0) 
        return round(pred[0][0][0])

    def predictCelebrity(self):
        open_cv_image = self.captured

        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

        if len(faces) == 0:
            print("No face detected in the captured image.")
            return None, "No Face Detected"

        x, y, w, h = faces[0]
        detected_face = open_cv_image[y:y+h, x:x+w]

        self.gender = self.genderClassifier(detected_face)

        resized_for_vgg = cv2.resize(detected_face, (224, 224))
        img_pixels = image.img_to_array(resized_for_vgg)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 127.5
        img_pixels -= 1
        captured_representation = self.VGGmodel.predict(img_pixels, verbose=0)[0, :] 

        df_patterns = None
        if self.gender == 1: 
            print("Predicted Gender: Male")
            df_patterns = self.male_patterns_df
        else:
            print("Predicted Gender: Female")
            df_patterns = self.female_patterns_df

        if df_patterns is None or df_patterns.empty:
            return None, "No patterns loaded for this gender."

        df_patterns_filtered = df_patterns.dropna(subset=['face_vector_raw'])
        df_patterns_filtered = df_patterns_filtered[df_patterns_filtered['face_vector_raw'].apply(
            lambda x: isinstance(x, np.ndarray) and x.size > 0
        )]

        if df_patterns_filtered.empty:
            print("No valid face vectors found in the filtered patterns dataframe for comparison.")
            return None, "No valid patterns for comparison."

        df_patterns_filtered.loc[:, 'similarity'] = df_patterns_filtered['face_vector_raw'].apply(
            lambda x: self.findCosineSimilarity(x, captured_representation)
        )
        
        min_similarity_row = df_patterns_filtered[df_patterns_filtered['similarity'] < 0.6].nsmallest(1, 'similarity')

        if not min_similarity_row.empty:
            instance = min_similarity_row.iloc[0]
            full_path = instance['Path']
            celeb_name = instance['Name']
            print(f"Closest match: {celeb_name} with similarity distance: {instance['similarity']:.4f}")
            return full_path, celeb_name
        else:
            print("No strong celebrity match found within threshold.")
            return None, "No Match Found"


# --- Main Execution ---
if __name__ == "__main__":
    CATEGORY_FOLDERS = ['Cricketers', 'English', 'Hindi', 'Kannada', 'Malayalam', 'Tamil', 'Telugu']

    if not os.path.exists('haarcascade_frontalface_default.xml'):
        print("Error: 'haarcascade_frontalface_default.xml' not found. Please download it and place it in the same directory.")
        sys.exit(1)
    
    male_df_initial, female_df_initial = prepare_datasets(CATEGORY_FOLDERS)

    vgg_model = loadVggFaceModel()

    male_patterns_df, female_patterns_df = generate_face_patterns(vgg_model, male_df_initial, female_df_initial)

    video_app = VideoCaptureApp()
    video_app.root.mainloop()