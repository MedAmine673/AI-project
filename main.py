import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import os
import tkinter as tk
from tkinter import Canvas, Button, Label

def load_data():
    categories = ["cat", "dog", "car", "airplane", "apple", "bicycle", "tree", "chair"]
    data, labels = [], []
    for idx, category in enumerate(categories):
        images = np.load(f"{category}.npy")[:10000]
        data.append(images)
        labels.append(np.full(len(images), idx))
    X = np.concatenate(data, axis=0).reshape(-1, 28, 28, 1) / 255.0
    y = np.concatenate(labels, axis=0)
    y = to_categorical(y, num_classes=len(categories))
    return X, y, categories
X, y, categories = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=128)
model.save("quickdraw_cnn.h5")
print("Model trained and saved as 'quickdraw_cnn.h5'!")

class DrawingApp:
    def __init__(self, root, model_path, categories):
        self.root = root
        self.root.title("AI Guesses Your Drawing")
        self.root.configure(bg='#DDEEFF')  # Light blue background
        self.canvas = Canvas(self.root, width=300, height=300, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.clear_button = Button(self.root, text="Erase", command=self.clear_canvas,
                                   font=("Arial", 14), width=15, height=2,
                                   bg="#FF6961", fg="white", activebackground="#D64545")
        self.clear_button.pack()
        self.predict_button = Button(self.root, text="Guess", command=self.predict_drawing,
                                     font=("Arial", 14), width=15, height=2,
                                     bg="#77DD77", fg="white", activebackground="#5CAB5C")
        self.predict_button.pack()
        self.result_label = Label(self.root, text="Draw and click on Guess!",
                                  font=("Arial", 16), fg="#444444", bg="#DDEEFF", pady=10)
        self.result_label.pack()
        self.image = np.zeros((300, 300), dtype=np.uint8)
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.categories = categories
        else:
            self.model = None
            self.result_label.config(text="Model not found. Train a model!")
    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x + 8, y + 8, fill='black', outline='black')
        cv2.circle(self.image, (x, y), 4, (255), -1)
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = np.zeros((300, 300), dtype=np.uint8)
    def predict_drawing(self):
        if self.model is None:
            self.result_label.config(text="Model not found")
            return
        resized_img = cv2.resize(self.image, (28, 28)) / 255.0
        resized_img = resized_img.reshape(1, 28, 28, 1)
        predictions = self.model.predict(resized_img)[0]
        top_prediction = np.argmax(predictions)
        result_text = f"Prediction : {self.categories[top_prediction]} \n Confidence : {predictions[top_prediction] * 100:.2f}%"
        self.result_label.config(text=result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root, "quickdraw_cnn.h5", categories)
    root.mainloop()