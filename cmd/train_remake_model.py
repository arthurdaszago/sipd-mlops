import os
import sys
import mlflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

num_other_findings_train_samples = sys.argv[1]

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')

# ================================================

other_findings_train_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_other_findings_images.npy'))
other_findings_train_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_other_findings_labels.npy'))

other_findings_train_images = other_findings_train_images[..., np.newaxis]

print('other_findings_train_images.shape: ', other_findings_train_images.shape)
print('other_findings_train_labels.shape: ', other_findings_train_labels.shape)

model_path = os.path.join(PATH_ROOT, 'model', 'cnn_model.h5')

# Carrega o modelo salvo
model = tf.keras.models.load_model(model_path)

# Remova a última camada do modelo
model.layers.pop()

# Numero de possíveis saidas
new_outputs = 4 
output = Dense(new_outputs, activation='softmax')(model.layers[-1].output)

# Crie um novo modelo com a saída alterada
new_model = Model(inputs=model.input, outputs=output)

# summary model structure
model.summary() 

# Compilando o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# early stop
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Treinando o modelo
history = model.fit(other_findings_train_images, other_findings_train_labels, epochs=25, validation_split=0.1, callbacks=[callback])

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

mlflow.run(uri='.', entry_point='experiments_remake', parameters={ 'num_train_samples': 1 })