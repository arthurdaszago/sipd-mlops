import os
import sys
import mlflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_samples = sys.argv[1]

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
sys.path.append(PATH_ROOT)

TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')
model_path = os.path.join(PATH_ROOT, 'model', 'cnn_model.h5')

from src.utils.shuffle import shuffle_in_order
from src.utils.laod_retrain_dataset import load_retrain_dataset

# ================================================

(covid_images, covid_labels), (normal_images, normal_labels), (pneumonia_images, pneumonia_labels), (other_findings_images, other_findings_labels) = load_retrain_dataset(num_samples)

retrained_images = np.vstack((covid_images, normal_images, pneumonia_images, other_findings_images))
retrained_labels = np.vstack((covid_labels, normal_labels, pneumonia_labels, other_findings_labels))

retrained_images, retrained_labels = shuffle_in_order(retrained_images, retrained_labels)

print('retrained_images.shape: ', retrained_images.shape, ', retrained_labels.shape: ', retrained_labels.shape)

# Carrega o modelo salvo
model = tf.keras.models.load_model(model_path)

# summary model structure
model.summary() 

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# early stop
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0.075)

# Treinando o modelo
history = model.fit(retrained_images, retrained_labels, batch_size=8, epochs=25, validation_split=0.1, callbacks=[callback])

# Plotando a acurácia de treino e validação ao longo das épocas
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.savefig(os.path.join(PATH_ROOT, 'stats', 'train', 'retrained_accuracy_graph.png'))

# Plotando a perda de treino e validação ao longo das épocas
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# save fig
plt.savefig(os.path.join(PATH_ROOT, 'stats', 'train', 'retrained_loss_graph.png'))

# Se você quiser salvar o modelo após o treinamento, você pode fazer isso:
model.save(os.path.join(PATH_ROOT, 'model', 'cnn_model_retrained.h5'))

mlflow.run(uri='.', entry_point='retest_model_retrained', parameters={ 'num_samples': num_samples + num_samples })