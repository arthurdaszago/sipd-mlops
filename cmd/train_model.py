import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')

sys.path.append(PATH_ROOT)

from src.utils.model import define_model

# ================================================

train_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_images.npy'))
train_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_labels.npy'))

print('train_images.shape: ', train_images.shape)
print('train_labels.shape: ', train_labels.shape)

model_path = os.path.join(PATH_ROOT, 'model', 'cnn_model.h5')

if os.path.exists(model_path):
    pass
else:
    # Criando o modelo
    model = define_model()

    model.summary() 

    # early stop
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0.075)

    print('train_images.shape: ', train_images.shape, ', train_labels.shape: ', train_labels.shape)

    # Treinando o modelo
    history = model.fit(train_images, train_labels, batch_size=4, epochs=25, validation_split=0.1, callbacks=[callback], verbose=1)

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