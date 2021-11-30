import torchio as tio
#https://colab.research.google.com/github/fepegar/torchio-notebooks/blob/main/notebooks/Data_preprocessing_and_augmentation_using_TorchIO_a_tutorial.ipynb

def addNoise(image,std=0.5): #channels, x,y
  im4d=image.unsqueeze(-1)
  standardize = tio.ZNormalization()  
  add_noise = tio.RandomNoise(std)
  standard = standardize(im4d)
  noisy = add_noise(standard)  
  return noisy.squeeze(-1)

