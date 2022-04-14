import librosa as l
mfcc_data = []
x, sr = l.load('D:/UIN/AI/Tugas Akhir AI/code/pop/pop.00000.wav', sr = 22050)
n_fft = 2048 #int(sr * 0.02)   # window length: 0.02 s
hop_length = 512 #n_fft // 2  # usually one specifies the hop length as a fraction of the window length
mfccs = l.feature.mfcc(x, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)



# check the dimensions
mfccs = mfccs.T
print(mfccs.shape)

num_mfcc_vectors_per_segment = 130

temp = 0
if len(mfccs) == num_mfcc_vectors_per_segment:
    temp += 1
    mfcc_data.append(mfccs.tolist())
    # print("{}, segment:{}".format(file_path, d+1))
print(temp)