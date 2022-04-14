import json
import numpy as np
import matplotlib.pyplot as plt
# DATA_PATH = DATASET_PATH = "data_music_new.json"

# with open(DATASET_PATH, "r") as fp:
#     data = json.load(fp)
# inputs = np.array(data["mfcc"])
# targets = np.array(data["labels"])
# # print(inputs.shape)


label = ["blues",
        "classic",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"]
min = [
    -0.8402405,
    -0.34362793,
    -0.77111816,
    -0.749176,
    -1.0,
    -0.6347351,
    -0.4850464,
    -1.0,
    -0.6341858,
    -0.960083]
max = [
    0.885376,
    0.31762695,
    0.77352905,
    0.73010254,
    0.96673584,
    0.6819763,
    0.49539185,
    0.9999695,
    0.60220337,
    0.97436523
]
x = range(1,11)
title_max = "Frekuensi Tertinggi Setiap Genre"
title_min = "Frekuensi Terendah Setiap Genre" 
plt.title(title_max)
plt.bar(x, max)
plt.xticks(x, label)
plt.tight_layout()
plt.show()