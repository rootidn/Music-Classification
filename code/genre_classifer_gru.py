import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import time
from datetime import datetime

DATA_PATH = DATASET_PATH = "data_music_new.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # ubah list ke np array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def prepare_datasets(test_size, validation_size):
    #load data
    x, y = load_data(DATASET_PATH)
    
    #buat train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    #buat train validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size)

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def build_model(input_shape):
    # membuat model rnn-lstm
    model = keras.Sequential()
    
    # make 2 lstm layer
    model.add(keras.layers.GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.GRU(64))
    model.add(keras.layers.BatchNormalization())
    
    #dense layer
    # model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def predict(model, x, y):
    x = x[np.newaxis, ...]

    prediction = model.predict(x) #x -> 3d array = {130, 13, 1}, butuh 4d = {1,130,13,1}
    #hasil prediction = [[0.1, 0.2],...]
    
    #extract prediction index dgn value terbesar
    predicted_index = np.argmax(prediction, axis=1)
    print("expected genre idx: {} prediksi genre idx: {}".format(y, predicted_index))

def plot_history(history):

    fig, axs = plt.subplots(2)

    #visualisasi akurasi
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    #visualisasi error
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    # Set waktu komputasi
    start = time.time()
    
    #buat train validation dan test sets
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    
    #buat CNN network
    input_shape = (x_train.shape[1], x_train.shape[2]) # [130, 13]
    model = build_model(input_shape)

    #compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #train CNN
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=100)

    # Set akhir waktu komputasi 
    end = time.time()
    # Proses menghitung waktu komputasi
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    # Hasil waktu komputasi
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


    #plot accuracy
    plot_history(history)

    #evaluasi CNN dengan test set
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("accuracy test sets adalah {}".format(test_accuracy))

    #membuat prediksi berdasarkan sample
    x = x_test[100]
    y = y_test[100]
    predict(model, x, y)
