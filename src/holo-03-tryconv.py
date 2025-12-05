#!/usr/bin/env python3

import os,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
def fixhololevel(pdata):
    # grayscale only
    if len(pdata.shape) > 2:
        data = np.float32(cv2.cvtColor(data,cv2.COLOR_BGR2GRAY))
    else:
        data=np.float32(pdata.copy())
    # normalize to 0..1
    data -= data.min()
    data /= data.max()
    # compute histogram
    datahst = np.histogram(data, bins=256, range=(0.0, 1.0))[0]
    # find most common value
    mostcommon = np.argmax(datahst)/256
    # shift so that most common value is at 0.5

    n = np.log(0.5) / np.log(mostcommon)
    data = data ** n

    return data

img_size = 512
fftpad = 128
fftp = 8
#holo = np.float32(cv2.imread("composites/holo/gds_output00006.png")/255)[...,0]
#kernel = np.float32(cv2.imread("onepix/holo/onepixel01.png")/255.0)[...,0]
#gt = np.float32(cv2.imread("composites/gt/gds_output00006.png")/255.0)[...,0]

# establish values that will not change based on holo params
xx = np.linspace(-img_size//2,img_size//2-1,img_size)
yy = np.linspace(-img_size//2,img_size//2-1,img_size)
x,y = np.meshgrid(xx,yy)
padding = np.ones([img_size+fftpad*2]*2)*0.5

holo = fixhololevel(np.float32(cv2.imread(sys.argv[1])/255)[...,0])
datahome, basename = os.path.split(sys.argv[1])
datahome, datalastdir = os.path.split(datahome)
# holo params
wavelen = 635.0e-9
dx = 1.12e-6
r = np.sqrt(x*x + y*y) * dx

render_number = 0

paddedholo = padding.copy()
paddedholo[fftpad:-fftpad,fftpad:-fftpad] = holo

#zees = np.arange(1e-3-wavelen*1600,6e-3+wavelen*1000,wavelen*500)

zees = np.exp(np.arange(np.log(0.0001),np.log(.01),0.025))

#zees = np.arange(0.001, 0.003, 0.0001) 
#zees = np.float32([0.0028])
#zees = np.float32((0.00328, 0.00328001))
zi = 0
paused = False

def makefig(zi):
    render_number = zi+1
    # these will change
    z = zees[zi]
    d = np.sqrt(r*r+z*z)

    kmask= 1.0 / (d+1)
    amp = -np.sin((d / wavelen) * np.pi*2)  * kmask
    amp = amp / 2.0 + 0.5


    paddedamp =padding.copy()
    paddedamp[fftpad:-fftpad,fftpad:-fftpad] = amp

    paddedLaplacian = padding.copy()
    pdh = padding.shape[0]//2

    paddedLaplacian[...] = 0
    paddedLaplacian[pdh-1:pdh+2, pdh-1:pdh+2] = (
        np.float32(
           [[ 0, 1, 0 ],
            [ 1, -4, 1 ],
            [ 0, 1, 0]]
        )
    )



    kernelFFT = np.fft.fft2(paddedamp)
    print(f"kernelFFT.shape={kernelFFT.shape} kernelFFT.dtype={kernelFFT.dtype}")
    print(f"np.real(kernelFFT).shape={np.real(kernelFFT).shape} np.real(kernelFFT).dtype={np.real(kernelFFT).dtype}")

    holoFFT = np.fft.fft2(paddedholo)

    LapFFT = np.fft.fft2(paddedLaplacian)

    prodFFT = kernelFFT * holoFFT

    edgesFFT = prodFFT * LapFFT * LapFFT 

    prodpadded = np.fft.fftshift(np.fft.ifft2(prodFFT))

    edgesPadded = np.fft.fftshift(np.fft.ifft2(edgesFFT))

    prod=prodpadded[fftp:-fftp,fftp:-fftp]
    edges = edgesPadded[fftp:-fftp,fftp:-fftp]
    print(prod[:5,:5])

    im = np.abs(prod)
    im = fixhololevel(im)

    edgesim = (np.abs(edges))
    edgesim = cv2.absdiff(edgesim,np.mean(edgesim))

    edgesblur = cv2.GaussianBlur(im, (0,0), sigmaX=3, sigmaY=3)
    edgesblur2 =cv2.GaussianBlur(im, (0,0), sigmaX=5, sigmaY=5)
    edgesdiff = cv2.absdiff(edgesblur,edgesblur2)
    edgesim = edgesdiff
    #edgesim[edgesim<np.quantile(edgesim,.99)]=0
    #alpha=0.8
    #edgesim =  (edgesim * (1.5+alpha) + ( edgesblur * (-0.5) ))
    cimage = im# * edgesim
    #cdiff = np.abs(cimage-0.5)
    #edgesim=cdiff
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(12,6)
    cimage=im #np.stack((im/2,im/2+edgesim/2,im/2+edgesim/2),axis=2)
    #ax[0].imshow(gt,cmap='binary_r')
    #ax[0].set_title('gt',fontsize=30)
    #ax[1].imshow(holo,cmap='binary_r')
    #ax[1].oset_title('holo',fontsize=30)
    print("******************")
    print(f"mkdir {datahome}/cimage")
    print(f"save to {datahome}/cimage/{basename}")
    os.makedirs(f"{datahome}/cimage", exist_ok=True)
    cv2.imwrite(f"{datahome}/cimage/{basename}",np.uint8(cimage*255))
    print("******************")
    ax[0].imshow(cimage,cmap='binary_r')
    ax[0].set_title('cimage',fontsize=30)
    ax[1].imshow(edgesim,cmap='binary_r')
    ax[1].set_title('edgesim',fontsize=30)
    suptitle = f"z = {z} [{zi}]"
    plt.suptitle(suptitle,fontsize=40)
    plt.tight_layout()
    os.makedirs(f"{datahome}/figs", exist_ok=True)
    figfile=f"{datahome}/figs/fig{render_number:03d}.png"
    plt.savefig(figfile)
    plt.close('all')
    if len(zees) == 1:
        sys.exit()
    cv2.imshow("holomagic", cv2.imread(figfile))
    k=cv2.waitKey(1) & 255
    print(f"k={k}")
    if k == 27:
        break
    if k == 32:
        paused = not paused
    if k == ord(','):
        zi = (zi-1) if zi > 0 else len(zees)-1
        paused = True
    if k == ord('.'):
        zi = (zi + 1) if zi < len(zees)-1 else 0
        paused = True
    if not paused:
        zi+=1
        if zi > len(zees)-1:
            zi = 0
            sys.exit(0)    


while True:
    makefig(zi)
