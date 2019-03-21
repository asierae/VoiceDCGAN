# VoiceDCGAN
Human Voice Cloner Generative Network

![training process](./gen/spectrograms_epoch.gif "training")


Requirements
---
- keras >= 2.1
- tensorflow >= 1.4
- librosa
- imutils
- Python 3



Dataset
--
Wav File

Preprocess
---
Wav --> wav chunks --> spectograms --> mel spectograms --> feed DCGAN


Training
--- 

![training process random](./gen/spectrograms_epoch2.gif "training2") 

Generate Sounds
---
mel spectograms --> spectograms --> wav file

Result
---




Credits
---
- The implementation of Deep Convolutional GAN from idea in https://arxiv.org/abs/1704.00028 
.
