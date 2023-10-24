import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


TRAIN_DATASET_PATH = None
if os.getenv('TRAIN_DATASET_PATH') is not None:
    TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')
else:
    TRAIN_DATASET_PATH = '/home/arthur/Documents/ifc/tc/code/sipd-mlops/datasets/train'

PATH_ROOT = None
if os.getenv('PATH_ROOT') is not None:
    PATH_ROOT = os.getenv('PATH_ROOT')
else:
    PATH_ROOT = '/home/arthur/Documents/ifc/tc/code/sipd-mlops'

train_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_images.npy'))
train_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_labels.npy'))


# Criando o modelo
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # 5 classes no total (avião, caminhão, navio, outros, sapo)
])

# Compilando o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# Plotando a acurácia de treino e validação ao longo das épocas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Plotando a perda de treino e validação ao longo das épocas
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()

# save fig
plt.savefig(os.path.join(PATH_ROOT, 'stats', 'train', 'accuracy_loss_graph.png'))

# Se você quiser salvar o modelo após o treinamento, você pode fazer isso:
model.save(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))