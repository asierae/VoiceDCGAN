from keras.models import Sequential,Model,load_model
from Wav2Spec2MelSpec import *

def gen_melspec2wav(samples,generator,step=0):
  if generator==None:
    generator = load_model("models/generator.h5")
  noise=np.random.uniform(-1.0, 1.0, size=[samples, 100])
  images = generator.predict(noise)
  for    filename = "gen/generated_"+str(step)+"_"+str(i)+".png"
 i in range(images.shape[0]):
    imageA = np.reshape(images[i], [64, 64,channels])
    if channels==1:
      imageA = imageA[:,:,0]#delete rgb
    pltsave.imsave(filename, imageA)
    imageG = cv2.imread(filename,1)
    imageG = imageG[:,:,0]#delete rgb
    #MEL2SPEC
    mel_inverted_spectrogram = mel_to_spectrogram(imageG, mel_inversion_filter,
                                                spec_thresh=spec_thresh,
                                                shorten_factor=shorten_factor)
    mel_inverted_spectrogram = ((mel_inverted_spectrogram/255)-1)*4#normalize 0/-4
    #SPEC2WAV
    inverted_mel_audio = invert_pretty_spectrogram(np.transpose(mel_inverted_spectrogram), fft_size = fft_size,
                                              step_size = step_size, log = True, n_iter = 10)
    inverted_mel_audio = inverted_mel_audio*100
    scipy.io.wavfile.write("/gen/gen_E"+str(step)+"_"+str(i)+".wav", sr, inverted_mel_audio)

print("Loading Model...")
generator = load_model("models/generator.h5")
print("Generating Wavs...")
gen_melspec2wav(10, generator)
print("Done!")