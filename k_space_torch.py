import torch
from torch.fft import fftn, ifftn
import numpy as np
def k_space_sampling(x,ss,idx):
    '''
    x: torch signal, of total lenght equal prod(ss).
    ss: signal shape
    idx: indexes to take the k-space measurements
    '''
    
    x=x.reshape(ss)
    X=fftn(x,s=ss)
    X=X.reshape((-1,1))
    return X[idx]

def adjoint(y,ss,idx):
    #Y=torch.zeros(ss)
    Y=np.zeros(ss,dtype=complex)
    Y=torch.from_numpy(Y)
    Y=Y.reshape((-1,1))
    Y[idx]=y
    Y=Y.reshape(ss)
    Y=ifftn(Y,ss)
    return Y.reshape((-1,1))

def l2_rec(img,idx):
    ss=img.shape
    y=k_space_sampling(img,ss,idx)
    rec=adjoint(y,ss,idx)
    return abs(rec.reshape(ss))

if __name__ == "__main__":

    import matplotlib.pyplot as plt 
    from PIL import Image
    from skimage.transform import resize
    from skimage.data import shepp_logan_phantom
    image = shepp_logan_phantom()    
    image=resize(image,(128,128))
    image=image.astype(np.float)/np.max(image[...])
    image=torch.from_numpy(image)
    #random sampling:
    numpoints=int(128*128/2)
    idx=np.random.permutation(np.arange(0, 128*128).tolist()).tolist()
    idx=idx[0:numpoints-1]
    p=np.zeros((128,128))
    p=p.reshape((-1,1))
    p[idx]=1
    p=p.reshape((128,128))
    y=k_space_sampling(image,(128,128),idx)
        
    #X_toview=np.log(1+abs(X))
    
    rec=adjoint(y,(128,128),idx)
    rec=rec.reshape((128,128))

    plt.subplot(1,4,1)
    plt.imshow(image)
    plt.subplot(1,4,2)
    plt.imshow(p)
    plt.subplot(1,4,3)
    plt.imshow(abs(rec))
    plt.subplot(1,4,4)
    print(type(idx))
    plt.imshow(abs(l2_rec(image,idx)))

    plt.show()
