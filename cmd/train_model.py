import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')

# ================================================

train_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_images.npy'))
train_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_labels.npy'))

train_images = train_images[..., np.newaxis]

print('train_images.shape: ', train_images.shape)
print('train_labels.shape: ', train_labels.shape)

model_path = os.path.join(PATH_ROOT, 'model', 'cnn_model.h5')

if os.path.exists(model_path):
    pass
else:
    # Criando o modelo
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(224, 224, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3,3), activation='relu'),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes no total (covid, normal ou pneumonia)
    ])

    model.summary() 

    # Compilando o modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Treinando o modelo
    history = model.fit(train_images, train_labels, epochs=5, validation_split=0.1)

    # Plotando a acurácia de treino e validação ao longo das épocas
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig(os.path.join(PATH_ROOT, 'stats', 'train', 'accuracy_graph.png'))

    # Plotando a perda de treino e validação ao longo das épocas
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # save fig
    plt.savefig(os.path.join(PATH_ROOT, 'stats', 'train', 'loss_graph.png'))

    # Se você quiser salvar o modelo após o treinamento, você pode fazer isso:
    model.save(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))