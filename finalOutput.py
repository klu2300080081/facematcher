#Run th following files first
#mFdatset.ipynb
#prototype-2.ipynb
#all files require jupyter notebook and the dataset should be required
#####################################################################
from tkinter import *
import cv2
from PIL import Image, ImageTk
import tensorflow as tf # This is necessary

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

from keras.preprocessing import image
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from keras import metrics 
from tensorflow.keras.utils import load_img
from PIL import Image

class VideoCaptureApp:
    def __init__(self):
        self.mae = metrics.mean_absolute_error  # Assign 'mae' to the function
        self.GCmodel = load_model('genderClassifier.h5', custom_objects={'mae': self.mae})
        self.VGGmodel = load_model("VGGnet.h5")
        self.root=Tk()
        self.root.configure(bg="blue")
        
        # Get screen width and height
        self.screen_width = self.root.winfo_screenwidth()-100
        self.screen_height = self.root.winfo_screenheight()-100
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+50+50")
        #frame dimensions
        self.frame_width=500
        self.frame_height=600

        #head
        self.headText=Label(self.root, text="Main Head",justify=CENTER)
        self.headText.pack(padx=5,pady=5)

        self.constructLeftFrame()
        self.caputure()
        self.constructRightFrame()
        #submit
        self.submit_button=Button(self.root, text="Capture", command=self.display, justify=CENTER)
        self.submit_button.pack(side=BOTTOM)
        

    def constructLeftFrame(self):
        #left frame
        self.leftFrame=Frame(self.root, width= self.frame_width, height=self.frame_height, bg="white")
        self.leftFrame.pack(side=LEFT,padx=50)
        # Prevent the frame from resizing to fit its contents
        self.leftFrame.pack_propagate(False)

        self.leftHead=Label(self.leftFrame,text="Left side",justify=CENTER)
        self.leftHead.pack(side=TOP,pady=5)

    def constructRightFrame(self):
        self.rightFrame=Frame(self.root, width= self.frame_width,height=self.frame_height, bg="white")
        self.rightFrame.pack(side=RIGHT,padx=50)
        #Left frame contents

        # Prevent the frame from resizing to fit its contents
        self.rightFrame.pack_propagate(False)


       
        #Right Frame
        self.rightHead=Label(self.rightFrame,text="Right side",justify=CENTER)
        self.rightHead.pack(side=TOP,pady=5)
        self.captured=None
        self.image_frame=self.rightFrame
        self.imglabel=Label(self.image_frame)
        self.imglabel.configure(image=self.captured)
        
        self.imglabel.pack(fill="both",expand=True)


    def caputure(self,video_source=0):
        self.video_frame = self.leftFrame
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        self.label = Label(self.video_frame)  # Place label inside the provided frame
        self.label.pack(fill="both", expand=True) # Make label fill the frame
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame.copy()
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.label.configure(image=self.photo)
            self.label.image = self.photo
        self.video_frame.after(15, self.update) # Use the frame's after method

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def display(self):
        self.captured = self.last_frame.copy()
        self.predCeleb=self.predictCelebrity()
        image = Image.open(self.predCeleb)
        photo = ImageTk.PhotoImage(image)
        self.imglabel.configure(image=photo)
        self.imglabel.image = photo

    def findCosineSimilarity(self,source_representation, test_representation):
        try:
            a = np.matmul(np.transpose(source_representation), test_representation)
            b = np.sum(np.multiply(source_representation, source_representation))
            c = np.sum(np.multiply(test_representation, test_representation))
            return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
        except:
            return 10 #assign a large value. similar faces will have small value.
        
    def genderClassifier(self,img_array):

        def get_image_features(img_array):
            # Use tf.keras.utils.load_img since it's already imported
            # Convert to grayscale using OpenCV
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            resized = resized.reshape(1, 128, 128, 1)
            resized = resized.astype("float32") / 255.0
            return resized
        features = get_image_features(img_array)
        pred = self.GCmodel.predict(features)
        return round(pred[0][0][0])





    def predictCelebrity(self):
        # Convert Tkinter PhotoImage to PIL → then to NumPy OpenCV format
        open_cv_image = self.captured

        # Predict gender using array directly
        self.gender = self.genderClassifier(open_cv_image)

        # Resize and preprocess for VGG model
        detected_face = cv2.resize(open_cv_image, (224, 224))
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 127.5
        img_pixels -= 1
        captured_representation = self.VGGmodel.predict(img_pixels)[0, :]

        # Load dataset
        if self.gender == 1:
            df = pd.read_pickle("Femalepatterns.pkl")
        else:
            df = pd.read_pickle("Malepatterns.pkl")

        # Compute cosine similarity
        df['similarity'] = df['face_vector_raw'].apply(
            self.findCosineSimilarity, test_representation=captured_representation
        )
        min_index = df[['similarity']].idxmin()[0]
        instance = df.loc[min_index]
        full_path = instance['Path']
        return full_path

        
        




video_app=VideoCaptureApp()
video_app.root.mainloop()




# print(f"{screen_width}x{screen_height}")












