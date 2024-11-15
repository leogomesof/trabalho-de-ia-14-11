import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Add, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Definição de Hiperparâmetros
imageRows, imageCols, cores = 32, 32, 3
batchSize = 64
numClasses = 10
epochs = 5

# Carrega o dataset CIFAR-10
(XTreino, yTreino), (XTeste, yTeste) = cifar10.load_data()

# Normaliza os dados
XTreino = XTreino / 255.0
XTeste = XTeste / 255.0
yTreino = to_categorical(yTreino, numClasses)
yTeste = to_categorical(yTeste, numClasses)

inputShape = (imageRows, imageCols, cores)

# Define a entrada do modelo
inputs = Input(shape=inputShape)

# Primeira camada convolucional
x = Conv2D(50, (5, 5), padding='same', activation='relu')(inputs)  # Mudamos para 50 filtros para coincidir com o bloco residual
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

# Primeiro bloco residual
shortcut = Conv2D(50, (1, 1), padding='same')(x)  # Ajuste dimensional para o shortcut
x = Conv2D(50, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(50, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x, shortcut])  # Adiciona o shortcut ao output do bloco residual
x = Activation('relu')(x)

# Segundo bloco residual
shortcut = x  # Não precisa de ajuste pois já estão em 50 filtros
x = Conv2D(50, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(50, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x, shortcut])  # Adiciona o shortcut ao output do bloco residual
x = Activation('relu')(x)

# Pooling
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

# Flatten e camada densa
x = Flatten()(x)
x = Dense(500, activation='relu')(x)

# Camada de saída
outputs = Dense(numClasses, activation='softmax')(x)

# Criação do modelo
model = Model(inputs=inputs, outputs=outputs)

# Compilação do modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
minhaResnetModel = model.fit(XTreino, yTreino, batch_size=batchSize, epochs=epochs, validation_data=(XTeste, yTeste))

# Avaliação do modelo
nomeDosRotulos = ["avião", "carro", "pássaro", "gato", "cervo", "cachorro", "sapo", "cavalo", "navio", "caminhão"]
predicao = model.predict(XTeste)
print(classification_report(yTeste.argmax(axis=1), predicao.argmax(axis=1), target_names=nomeDosRotulos))

# Plot do Gráfico de Acurácia
plt.plot(minhaResnetModel.history['accuracy'], 'o-')
plt.plot(minhaResnetModel.history['val_accuracy'], 'x-')
plt.legend(['Acurácia no Treinamento', 'Acurácia na Validação'], loc=0)
plt.title('Treinamento e Validação - Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.show()
