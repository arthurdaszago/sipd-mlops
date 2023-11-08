import os
import gc
import sys
import mlflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


print('sys.argv: ', sys.argv)

tax_samples = float(sys.argv[1])

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
sys.path.append(PATH_ROOT)

TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')
model_path = os.path.join(PATH_ROOT, 'model', 'cnn_model.h5')
retrained_model_path = os.path.join(PATH_ROOT, 'model', 'cnn_model_retrained.h5')

from src.utils.shuffle import shuffle_in_order
from src.utils.laod_retrain_dataset import load_retrain_dataset

# ================================================

(covid_images, covid_labels), (normal_images, normal_labels), (pneumonia_images, pneumonia_labels), (other_findings_images, other_findings_labels) = load_retrain_dataset(tax_samples)

retrained_images = np.concatenate((covid_images, normal_images, pneumonia_images, other_findings_images), axis=0)
retrained_labels = np.concatenate((covid_labels, normal_labels, pneumonia_labels, other_findings_labels), axis=0)

retrained_images, retrained_labels = shuffle_in_order(retrained_images, retrained_labels)

print('retrained_images.shape: ', retrained_images.shape, ', retrained_labels.shape: ', retrained_labels.shape)

if os.path.exists(retrained_model_path):
  pass
else:
  # Carrega o modelo salvo
  model = tf.keras.models.load_model(model_path)

  # summary model structure
  model.summary() 

  # Compilando o modelo
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # early stop
  callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0.075)

  # Supondo que seus rótulos são inteiros de 0 a 3
  retrained_labels_one_hot = tf.keras.utils.to_categorical(retrained_labels, num_classes=4)

  # Treinando o modelo
  history = model.fit(retrained_images, retrained_labels_one_hot, batch_size=8, epochs=10, validation_split=0.1, callbacks=[callback], verbose=1)

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

mlflow.run(uri='.', entry_point='retest_model_retrained', parameters={ 'tax_samples': tax_samples })