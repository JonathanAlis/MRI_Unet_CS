import numpy as np 
import math
import matplotlib.pyplot as plt

def vertical_subsampling(s, rate=2, low_freq=12/100, show=False):
    z=np.zeros(s)
    z[0::rate,:]=1
    lpi=(math.floor(-low_freq/2*s[0]),math.ceil(low_freq/2*s[0])) #lowpass idx
    if lpi[0]!=0:
        z[:lpi[1],:]=1
        z[lpi[0]:,:]=1

    if show:
        plt.imshow(np.fft.ifftshift(z))
        plt.show()
    
    z=z.reshape((-1,1))
    idx= [i for i, x in enumerate(z) if x == 1]    
    
    
    
    return idx

if __name__ == "__main__":
    idx=vertical_subsampling((128,128), rate=4, low_freq=12/100, show=True)
    print(idx)
    y=np.zeros((128,128))
    y=y.reshape((-1,1))
    y[idx]=1
    y=y.reshape((128,128))
    y=np.fft.fftshift(y)
    plt.imshow(y)
    plt.show()
