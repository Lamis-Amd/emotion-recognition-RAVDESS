# emotion-recognition-RAVDESS Description
In this notebook: building a multimodal deep neural network for emotion detection using tf.keras. You have to work with the RAVDESS dataset, which contains short (~4 seconds long) video clip recordings of speakers, who are acting the different emotions through 2 sentences. We will extract and combine RGB frames with MFCCs and utilize both video and audio information sources to achieve a better prediction.



## Prepare dataset
Download the RAVDESS dataset https://zenodo.org/record/1188976#.X5g53OLPw2w The dataset is available here as well: http://nipg1.inf.elte.hu:8765 ravdess.zip contains all of the mp4 clips. The labels are in the file names. (classification task)

Preprocess the data.

Remove the silence parts from the beginning and the end of video clips. (Tips: ffmpeg filters)
Audio representation:
Extract the audio from the video. (Tips: ffmpeg)
Extract 24 Mel Frequency Cepstral Coefficients from the audio. (Tips: use librosa.)
Calculate the mean number of (spectral) frames in the dataset.
Standardize the MFCCs sample-wise. (Tips: zero mean and unit variance)
Use pre-padding (Note: with 0, which is also the mean after standardization) to unify the length of the samples.
Audio representation per sample is a tensor with shape (N,M,1) where N is the number of coefficients (e.g. 24) and M is the number of audio frames.
Visual representation:
Extract the faces from the images. (Tips: You can use the cv2.CascadeClassifier, or the DLIB package to determine facial keypoints, or MTCNN to predict bounding boxes.)
Resize the face images to 64x64. (Tips: You can use lower/higher resolution as well.)
Subsample the frames to reduce complexity (6 frames/video is enough).
Apply data augmentation, and scaling [0, 1].
Video representation per sample is a tensor with shape (F,H,W,3) where F is the number of frames (e.g. 6), H and W are the spatial dimensions (e.g. 64).
Ground truth labels:
There are 8 class labels. However, Class 1 (Neutral) and Class 2 (Calm) are almost the same. It is a commonly used practice to merge these two classes. Combine them to reduce complexity.
(Optional) Use one-hot-encoding with categorical_crossentropy loss later on, or keep them between [0, 6] and use sparse_categorical_crossentropy loss. It's up to you.
Split the datasets into train-valid-test sets. Samples from the same speaker shouldn't appear in multiple sets. (Example split using speaker ids: 1-17: train set, 18-22: validation set, 23-24: test set)

Create a generator, which iterates over the audio and visual representations. (Note: the generator should produce a tuple ([x0, x1], y), where x0 is the audio, x1 is the video representation, y is the ground truth.

Print the size of each set, plot 3 samples: frames, MFCCs and their corresponding emotion class labels. (Tips: use librosa for plotting MFCCs)

Alternative considerations. They may require additional steps:

You can use Mean (axis=1) MFCCs vectors to further reduce complexity. Input of the corresponding subnetwork should be modified to accept inputs with shape (N, 1).
You can use log-melspectrograms as well. Note, that raw spectrograms are displaying power. Mel scale should be applied on the frequency axis, and log on the third dimension (decibels are expected). You can use librosa for that (librosa.feature.melspectrogram, librosa.power_to_db)
A better evaluation procedure here is the LOO (Leave-One-Out) cross-validation, however it can be costy.



## Create Model
Create the audio subnetwork.

BLSTM (64 units, return sequences) + Dropout 0.5 + BLSTM (64 units) + Dense (128 units, ReLU)

Create the visual subnetwork
Choose a visual backbone, which is applied frame-wise (Tips: use TimeDistributed Layer for this):
VGG-like architecture (Conv2D + MaxPooling blocks)
ResNet / Inception architecture (Residual blocks, Inception cells)
You can try other configurations, better submodels (like 3D convolution nets). Have a reason for your choice!
Apply Max pooling over the time dimension to reduce complexity (or use GRU or LSTM for better temporal modelling)

##Model fusion:
Concatenate the final hidden representations of the audio and visual subnetwork.
Apply fully connected layers on it (256 units, ReLU), then an another dense layer (7 units, softmax).
You can feed multiple inputs to the Model using a list: model = tf.keras.models.Model(inputs=[input_audio, input_video], outputs=output)
