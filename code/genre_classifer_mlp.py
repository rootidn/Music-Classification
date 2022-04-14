# TODO : fix overfitting (model training set bagus tapi test set jelek)
# hypo = O dpt terjadi karena data dikit tapi maksa predict jadinya error besar

# step1 : melihat perubahan akurasi dan error di setiap epochs
# beberapa cara mengatasi overfitting:
    # - simpler architecture = hapus layer, mengurangi neuron 
    #       (semakin komplex semakin besar overvitting)
    # - audio data augmentation = 
            # (semakin banyak data semakin bagus model)
            # membuat data tambahan untuk training set,
            # caranya mentranformasikan audio file dgn:
            #  - pitch shifting (naik-turunkan pitch)
            #  - time streching (ubah speed)
            #  - menambahkan background noise
    # - early stopping
            # menghentikan epoch pada saat perbedaan error train & set 
            # pada jarak tertentu sebelum terjadi overfitting
    # - dropout (*)
            # secara random menonaktifkan neuron saat training
            # meningkatkan network robustness 
            # karena network dpt fokus ke neuron yg lebih sedikit
            # probabilitas droupout antra 0.1-0.5
    # - reguralization (*)
            # menambahkan penalty pada error function
            # menghukum weight yg terlalu besar
            # jenis
            # - L1: mengurangi value dr weight, 
            #       bagus jika ada oulier, bagus jika modle simple
            # - L2: mengurangi kuadrad value dr wight,
            #       jelek jk ada outlier, bagus jika model kompleks  

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.python.ops.gen_resource_variable_ops import assign_add_variable_op
import matplotlib.pyplot as plt
# pustaka untuk waktu komputasi
import time
from datetime import datetime

# DATASET_PATH = "D:\\UIN\\AI\\Tugas Akhir AI\\code\\data_music_new.json"
DATASET_PATH = "data_music_new.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # ubah list ke np array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

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

def predict(model, x, y):
    x = x[np.newaxis, ...]

    prediction = model.predict(x) #x -> 3d array = {130, 13, 1}, butuh 4d = {1,130,13,1}
    #hasil prediction = [[0.1, 0.2],...]
    
    #extract prediction index dgn value terbesar
    predicted_index = np.argmax(prediction, axis=1)
    print("expected genre idx: {} prediksi genre idx: {}".format(y, predicted_index))

# multiclass(10) classification
if __name__ == "__main__":
    # Set waktu komputasi
    start = time.time()

    #memuat data
    inputs, targets = load_data(DATASET_PATH)

    # membagi data jadi training dan testing
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.3)
    
    # membuat network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
            # Dlatten : membuat multiD array menjadi flat
            # D1 = interval, D2 = value dlm interval
            # inputs adalah 3D array, yg ke 0 itu semuanya
        
        # 1st hidden layer
        keras.layers.Dense(512, activation="relu"),#, kernel_regularizer=keras.regularizers.L2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
            # Rectified Linear Unit : (higher convergance, fix vanishing gradient issue)
        
        # 2st hidden layer
        keras.layers.Dense(256, activation="relu"),#, kernel_regularizer=keras.regularizers.L2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        # 3st hidden layer
        keras.layers.Dense(64, activation="relu"),#, kernel_regularizer=keras.regularizers.L2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation="softmax")
            #sofmax = ???
    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        # Adam = ???
    model.compile(optimizer=optimizer, 
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    model.summary()
    
    # train network
    history = model.fit(inputs_train, targets_train, 
        validation_data=(inputs_test, targets_test),
        epochs=100,
        batch_size=32)

    # tipe batch:
        # stokastik = menghitung gradient dlm 1 sample (cepat, tdk akurat)
        # full batch = menghitung gradient di semua training set (lama, akurat)
        # mini-batch = menghitung gradien pd subset dataset (seimbang)

    
    # Set akhir waktu komputasi 
    end = time.time()
    # Proses menghitung waktu komputasi
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    # Hasil waktu komputasi
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    # visualisasi akurasi dan error pd setiap epochs
    plot_history(history)

# inputs_train, inputs_test, targets_train, targets_test
    #evaluasi CNN dengan test set
    test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
    print("accuracy test sets adalah {}".format(test_accuracy))

    #membuat prediksi berdasarkan sample
    x = inputs_test[100]
    y = targets_test[100]
    predict(model, x, y)