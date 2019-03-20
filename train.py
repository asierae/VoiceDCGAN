from Wav2Spec2MelSpec import *
from pydub import AudioSegment
from pydub.utils import make_chunks
from imutils import paths
from keras.models import load_model
from imutils import paths
from ModelDCGAN import DCGAN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltsave
import glob
import math
import random
import os
import time
import cv2


def generate_specs():
  wav_spectrogram = None
  print("[INFO] Generating Spectograms from chunks...")
  wavPaths = sorted(list(paths.list_files("wav")))
  random.seed(22)
  random.shuffle(wavPaths)
  #MEL SPECTOGRAMS
  for wav in wavPaths:
    rate, data = wavfile.read(wav)
    data = butter_bandpass_filter(data, lowcut, highcut, rate, order=1)
    # Only use a short clip for our demo
    if np.shape(data)[0]/float(rate) > 10:
      print("?")
      data = data[0:rate*10]
    #NORMAL SPECTOGRAM
    wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size = fft_size, 
                                   step_size = step_size, log = True, thresh = spec_thresh)
    
    pltsave.imsave("spectograms/specs/spec_"+wav.split("\\")[1]+".png",wav_spectrogram)
    #fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,5))
    #cax = ax.matshow(np.transpose(wav_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    #cax.write_png("specs/spec_"+wav.split("/")[1]+".png")
    

    mel_spec = make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor)
    pltsave.imsave("spectograms/specs_mel/m_spec_"+wav.split("\\")[1]+".png",mel_spec)
    #cax = ax.matshow(mel_spec, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    #cax.write_png("specs_mel/m_spec_"+wav.split("/")[1]+".png")

def wav2chunks(wav,chunk_length_ms=2000):
	print("[INFO] Sampling Wav in chunks...")
	myaudio = AudioSegment.from_file(wav , "wav") 
	chunk_length_ms = chunk_length_ms # pydub calculates in millisec
	chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

	#Export all of the individual chunks as wav files

	for i, chunk in enumerate(chunks):
	    chunk_name = "wav/chunk{0}.wav".format(i)
	    print ("exporting", chunk_name)
	    chunk.export(chunk_name, format="wav")


def load_specs():
  data = []
  print("[INFO] Loading Data...")
  imagePaths = sorted(list(paths.list_images(PATH)))
  random.seed(42)
  random.shuffle(imagePaths)
  # loop over the input images
  for imagePath in imagePaths:
    #image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    image = cv2.imread(imagePath,0)
    if IMG_SIZE>687:
      image=image[0:688,:,]#crop 688x1024-688x688  DESCOMENTAR SI 688
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    data.append(image)
  #normalize data for bestter optimization 0-1
  data = np.array(data, dtype="float") / 255.0
  print(len(data))
  return data

def gen_melspec2wav(samples,generator,step=0):
  if generator==None:
    generator = load_model("models/generator.h5")
  noise=np.random.uniform(-1.0, 1.0, size=[samples, 100])
  images = generator.predict(noise)
  for i in range(images.shape[0]):
    filename = "gen/generated_"+str(step)+"_"+str(i)+".png"
    imageA = np.reshape(images[i], [64, 64,channels])
    if channels==1:
      imageA = imageA[:,:,0]#delete rgb
    pltsave.imsave(filename, imageA)
    imageG = cv2.imread(filename,1)
    imageG = imageG[:,:,0]#delete rgb
    #imagePaths = [cv2.imread(file,1) for file in glob.glob('turin/generated*.png')]
    #MEL2SPEC
    mel_inverted_spectrogram = mel_to_spectrogram(imageG, mel_inversion_filter,
                                                spec_thresh=spec_thresh,
                                                shorten_factor=shorten_factor)
    mel_inverted_spectrogram = ((mel_inverted_spectrogram/255)-1)*4#normalize 0/-4
    #SPEC2WAV
    inverted_mel_audio = invert_pretty_spectrogram(np.transpose(mel_inverted_spectrogram), fft_size = fft_size,
                                              step_size = step_size, log = True, n_iter = 10)
    inverted_mel_audio = inverted_mel_audio*100
    scipy.io.wavfile.write("gen/gen_E"+str(step)+"_"+str(i)+".wav", sr, inverted_mel_audio)

def train(train_steps=2000, batch_size=256, save_interval=100):
    noise_input = None
    if save_interval>0:
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    for i in range(train_steps):
        images_train = x_train[np.random.randint(0,x_train.shape[0], size=batch_size), :, :, :]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        images_fake = generator.predict(noise)
        #print("Images train shape= ",images_train.shape)
        #print("Images noise shape= ",images_fake.shape)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0
        d_loss = discriminator.train_on_batch(x, y)

        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        a_loss = adversarial.train_on_batch(noise, y)
        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)
        if save_interval>0:
            if (i+1)%save_interval==0:
                adversarial.save("models/adversarial"+str(i)+".h5")
                discriminator.save("models/discriminator"+str(i)+".h5")
                generator.save("models/generator"+str(i)+".h5")
                plot_images(save2file=True, samples=noise_input.shape[0],\
                    noise=noise_input, step=(i+1))
                gen_melspec2wav(samples=6,generator=generator,step=i)

def plot_images(save2file=False, fake=True, samples=16, noise=None, step=0):
    filename = 'gen/gen.png'
    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "gen/samples_%d.png" % step
        images = generator.predict(noise)

    else:
        i = np.random.randint(0, x_train.shape[0], samples)
        images = x_train[i, :, :, :]

    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        if channels==1:
          image=images[i, :, :, 0]
        else:
          image = images[i, :, :, :]
          image = np.reshape(image, [IMG_SIZE, IMG_SIZE,channels])
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()

### Spec Parameters ###
fft_size = 2048 # window size for the FFT
step_size = fft_size//16 # distance to slide along the window (in time)
spec_thresh = 4 # threshold for spectrograms (lower filters out more noise)
lowcut = 500 # Hz # Low cut for our butter bandpass filter
highcut = 15000 # Hz # High cut for our butter bandpass filter
sr = 44100 #44100 Hz,puedo usar highcut
# For mels
n_mel_freq_components = 64 # number of mel frequency channels
shorten_factor = 10 # how much should we compress the x-axis (time)
start_freq = 300 # Hz # What frequency to start sampling our melS from 
end_freq = 8000 # Hz # What frequency to stop sampling our melS from 
# Generate the mel filters
mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)
### DCGAN Parameters ###
PATH="spectograms/specs_mel"#specs_mel=64x67 specs=1024x688(crop a 688x688) --dcgan 80x80
IMG_SIZE=64#688
EPOCHS=2000
BATCH_SIZE=128
LR=0.0005#learning rate for optimizer
channels = 1
data=[]



#Prepare Data
wav2chunks("YOURWAVFILE.wav")
generate_specs()
data = load_specs()
print(data[0].shape)

#Build Model
x_train = data.reshape(-1, IMG_SIZE,IMG_SIZE, channels).astype(np.float32)
myDCGAN = DCGAN(IMG_SIZE,IMG_SIZE,channels)#64x64 grayscale mel specs
discriminator = myDCGAN.discriminator_model()
adversarial = myDCGAN.adversarial_model()
generator = myDCGAN.generator()

#TRAIN
train(train_steps=EPOCHS, batch_size=BATCH_SIZE, save_interval=200)
plot_images(fake=True)
plot_images(fake=False, save2file=True)