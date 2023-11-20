import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')

sys.path.append(PATH_ROOT)

# from src.utils.model import define_model

# ================================================


model_path = os.path.join(PATH_ROOT, 'model', 'cnn_model.h5')

if not os.path.exists(model_path):
    train_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_images.npy'))
    train_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_labels.npy'))

    print('train_images.shape: ', train_images.shape)
    print('train_labels.shape: ', train_labels.shape)
    
    layers = tf.keras.layers

    # Criando o modelo
    model = tf.keras.models.Sequential()

    # Define o input shape
    input_shape = (224, 224, 3)  # Ajuste o input_shape conforme necessário

    # Adiciona uma camada de convolução baseada no EfficientNetB4
    model.add(tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    ))

    # Adiciona uma camada de Global Average Pooling 2D
    model.add(layers.GlobalAveragePooling2D(name='avg_pool'))

    # Adiciona uma camada de Batch Normalization
    model.add(layers.BatchNormalization(name='batch_1'))

    # Adiciona uma camada de Dropout
    model.add(layers.Dropout(rate=0.2, name='dropout_1'))

    # Adiciona uma camada densa com 1024 unidades e ativação ReLU
    model.add(layers.Dense(1024, activation='relu', name='dense_1'))

    # Adiciona uma camada de Dropout
    model.add(layers.Dropout(rate=0.2, name='dropout_2'))

    # Adiciona uma camada densa com 512 unidades e ativação ReLU
    model.add(layers.Dense(512, activation='relu', name='dense_2'))

    # Adiciona uma camada de Dropout
    model.add(layers.Dropout(rate=0.2, name='dropout_3'))

    # Adiciona uma camada densa com 64 unidades e ativação ReLU
    model.add(layers.Dense(64, activation='relu', name='dense_3'))

    # Adiciona uma camada de Dropout
    model.add(layers.Dropout(rate=0.2, name='dropout_4'))

    # Adiciona a camada de saída com 4 unidades e ativação softmax
    model.add(layers.Dense(4, activation='softmax', name='pred_layer'))

    # Compile o modelo com otimizador, função de perda e métrica
    optimizer = tf.keras.optimizers.Adam()  # Ajuste a taxa de aprendizado conforme necessário

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary() 

    # early stop
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0.075)

    print('train_images.shape: ', train_images.shape, ', train_labels.shape: ', train_labels.shape)

    # Supondo que seus rótulos são inteiros de 0 a 3
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=4)

    # Treinando o modelo
    history = model.fit(train_images, train_labels_one_hot, batch_size=8, epochs=10, validation_split=0.1, callbacks=[callback], verbose=1)

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