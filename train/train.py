
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Create the 'model' directory if it doesn't exist (safety check)
if not os.path.exists("../model"):
    os.makedirs("../model")

def train_eye_disease_model():
    print("\n--- Starting Training: Eye Disease Classifier ---")
    
    # Check if dataset exists
    if not os.path.exists("eye_dataset/"):
        print("Error: 'eye_dataset/' folder not found in current directory.")
        return

    # 1. Load Data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "eye_dataset/",
        image_size=(128, 128),
        batch_size=32,
        shuffle=True
    )

    # Get class names
    class_names = train_ds.class_names
    print(f"Eye Disease Classes found: {class_names}")

    # 2. Build Model
    model = models.Sequential([
        layers.InputLayer(input_shape=(128, 128, 3)),
        # Normalization Layer: Converts 0-255 -> 0-1
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])

    # 3. Compile & Train
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(train_ds, epochs=10)

    # 4. Save Model
    save_path = "../model/eye_model.h5"
    model.save(save_path)
    print(f"✅ Eye model saved to {save_path}")


def train_drowsiness_model():
    print("\n--- Starting Training: Drowsiness Detector ---")

    # Check if dataset exists
    if not os.path.exists("drowsy_dataset/"):
        print("Error: 'drowsy_dataset/' folder not found in current directory.")
        return

    # 1. Load Data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "drowsy_dataset/",
        image_size=(64, 64),
        batch_size=32,
        shuffle=True
    )
    
    print(f"Drowsiness Classes found: {train_ds.class_names}")

    # 2. Build Model
    model = models.Sequential([
        layers.InputLayer(input_shape=(64, 64, 3)),
        # Normalization Layer
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    # 3. Compile & Train
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(train_ds, epochs=5)

    # 4. Save Model
    save_path = "../model/drowsy_model.h5"
    model.save(save_path)
    print(f"✅ Drowsiness model saved to {save_path}")

if __name__ == "__main__":
    # Run both training functions
    train_eye_disease_model()
    train_drowsiness_model()

'''
import tensorflow as tf

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(224,224),
    batch_size=32
)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(dataset, epochs=5)
model.save("model.h5")

print("✅ Model trained and saved as model.h5")
'''