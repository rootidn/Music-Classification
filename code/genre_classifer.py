#music genre spesification using multi layer perception
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.python.ops.gen_resource_variable_ops import assign_add_variable_op

# DATASET_PATH = "D:\\UIN\\AI\\Tugas Akhir AI\\code\\data_music_new.json"
DATASET_PATH = "data_music_new.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # ubah list ke np array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

# multiclass(10) classification
if __name__ == "__main__":
    
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
        keras.layers.Dense(512, activation="relu"),
            # Rectified Linear Unit : (higher convergance, fix vanishing gradient issue)
        # 2st hidden layer
        keras.layers.Dense(256, activation="relu"),
        
        # 3st hidden layer
        keras.layers.Dense(64, activation="relu"),

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
    model.fit(inputs_train, targets_train, 
        validation_data=(inputs_test, targets_test),
        epochs=50,
        batch_size=32)

    # tipe batch:
        # stokastik = menghitung gradient dlm 1 sample (cepat, tdk akurat)
        # full batch = menghitung gradient di semua training set (lama, akurat)
        # mini-batch = menghitung gradien pd subset dataset (seimbang)

