import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model("long_hair_prediction_model.h5")

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_gender(img_path, age):
    img = preprocess_image(img_path)
    pred = model.predict(img)[0][0]
    is_long_hair = pred > 0.5

    if 20 <= age <= 30:
        return "Female" if is_long_hair else "Male"
    else:
        return "Female" if pred < 0.5 else "Male"

class GenderApp:
    def __init__(self, root):
        self.root = root
        root.title("Hair-Length Gender Classifier")

        self.label = Label(root, text="Upload Image and Enter Age:")
        self.label.pack()

        self.upload_btn = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.age_entry = tk.Entry(root)
        self.age_entry.pack()
        self.age_entry.insert(0, "Enter Age")

        self.predict_btn = Button(root, text="Predict Gender", command=self.predict)
        self.predict_btn.pack()

        self.result_label = Label(root, text="", font=("Arial", 16))
        self.result_label.pack()

        self.img_label = Label(root)
        self.img_label.pack()

        self.img_path = None

    def upload_image(self):
        self.img_path = filedialog.askopenfilename()
        img = Image.open(self.img_path)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def predict(self):
        try:
            age = int(self.age_entry.get())
            gender = predict_gender(self.img_path, age)
            self.result_label.config(text=f"Predicted Gender: {gender}")
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")

root = tk.Tk()
app = GenderApp(root)
root.mainloop()
