import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# Function to load and preprocess the dataset
def load_dataset(dataset_path):
    image_size = (224, 224)
    data = []
    labels = []
    

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                image = cv2.imread(os.path.join(folder_path, file))
                image = cv2.resize(image, image_size)
                image = image.astype("float") / 255.0
                data.append(image)
                labels.append(folder)

    return np.array(data), np.array(labels)

# Define a function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Load and preprocess the dataset
print("Loading and preprocessing the dataset...")
dataset_path = 'Dataset'
data, labels = load_dataset(dataset_path)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Load the pre-trained VGG16 model as the base
print("Loading pre-trained VGG16 model...")
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Add additional layers on top of VGG16
print("Adding additional layers on top of VGG16...")
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(labels)), activation='softmax')
])

# Compile the model
print("Compiling the model...")
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
callbacks = [EarlyStopping(patience=5, monitor='val_loss'), ModelCheckpoint('cnn_model.h5', save_best_only=True, monitor='val_loss')]
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=callbacks)

# Evaluate the model
print("Evaluating the model...")
train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Save the model and label encoder
print("Saving the model and label encoder...")
model.save('best_model.h5')
joblib.dump(label_encoder, 'label_encoder.pkl')
