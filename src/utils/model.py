import tensorflow as tf
from tensorflow.keras import layers, models

def define_model():
        # Define o modelo Sequential
    model = models.Sequential()

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

    return model