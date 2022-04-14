import matplotlib.pyplot as plt
import librosa
import numpy as np
plt.rcParams.update({'font.size': 12})

def visualisasi_audio():
    path_label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    song_label = []
    for p in path_label:
        song_label.append(p+'.00000.wav')
    fignum = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    plt.figure(figsize=(14,18))    
        
    for i in fignum:
        data, sr = librosa.load('D:/UIN/AI/Tugas Akhir AI/code/'+ path_label[i-1]+'/'+ song_label[i-1])
        plt.subplot(5, 2, i)
        plt.plot(data)
        m = np.array(data)
        max = m.max()
        min = m.min()
        # print(max)
        print(min)
        # plt.xlabel('sec')
        plt.ylabel('Hz', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('Visualisasi genre music ' + path_label[i-1], fontsize=10)
                    
#     plt.suptitle(title, fontsize=20, y=0.91)
    # plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.5)
    # plt.show()

if __name__ == '__main__':
    visualisasi_audio()