#!/usr/bin/env python3

import os,sys
import cv2
import numpy as np
#import matplotlib.pyplot as plt
import cupy
import cupy as cp
import context
import pyholoscope as pyh



def fixhololevel(pdata):
    data = None
    # grayscale only
    if len(pdata.shape) > 2:
        data = cp.array((pdata[...,1]).astype(cp.float32))
        # data = cp.float32(cv2.cvtColor(pdata,cv2.COLOR_BGR2GRAY))
    else:
        data=pdata.copy()
    if type(data) is np.ndarray:
        data = cp.array(data)
    data = cp.abs(data) 
    print(type(data))
    # normalize to 0..1
    data -= data.min()
    data /= data.max()
    # compute histogram
    print(type(data))
    datahst = cp.histogram(data, bins=256, range=(0.0, 1.0))[0]
    # find most common value
    mostcommon = np.argmax(datahst)/256
    # shift so that most common value is at 0.5

    n = np.log(0.5) / np.log(mostcommon)
    data = data ** n

    return data


img_size = 512
fftpad = 0
fftp = 8
#holo = np.float32(cv2.imread("composites/holo/gds_output00006.png")/255)[...,0]
#kernel = np.float32(cv2.imread("onepix/holo/onepixel01.png")/255.0)[...,0]
#gt = np.float32(cv2.imread("composites/gt/gds_output00006.png")/255.0)[...,0]

# establish values that will not change based on holo params
pimgsize = img_size + fftpad*2
xx = cp.linspace(-pimgsize//2,pimgsize//2-1,pimgsize)
yy = cp.linspace(-pimgsize//2,pimgsize//2-1,pimgsize)
x,y = cp.meshgrid(xx,yy)
padding = cp.ones([img_size+fftpad*2]*2)*0.5
padding = padding + 0j


# holo params
wavelen = 650.0e-9
dx = 1.12e-6
r = cp.sqrt(x*x + y*y) * dx

zees = np.arange(wavelen*10, 0.01, wavelen) 

def makefig(paddedholo, zi, imgptr):
    # these will change
    global zees, r
    z = zees[zi]
    d = cp.sqrt(r*r+z*z)

    print (f"\n****** [{zi}/{len(zees)}] z={z:8.60f} ******")

    # Create an instance of the Holo class
    holo = pyh.Holo(
        mode=pyh.INLINE,  # For inline holography
        wavelength=wavelen,  # Light wavelength, m
        pixel_size=dx,  # Hologram physical pixel size, m
    #    background=background,  # To subtract the background
        depth=zees[zi],
        invert=True,
        cuda=True
    )  # Distance to refocus, m

    
    # Refocus
    cimage = holo.process(paddedholo.get())
    return cimage

datahome,_ = os.path.split(sys.argv[1])
datahome,_ = os.path.split(datahome)
### per image execution here ###
imgptr = 1
zi = 0
print(f"CURPOS file: {datahome}/curpos.csv")

if os.path.exists(f"{datahome}/curpos.csv"):
    pos = open(f"{datahome}/curpos.csv","r").read().split(',')
    print(f"CURPOS={pos}")
    imgptr = int(pos[0])
    zi = int(pos[1])

escaped = False
while not escaped:
    if len(sys.argv) < 2:
        break
    hologram = holo.

    cp.array(cv2.medianBlur(cv2.imread(sys.argv[imgptr]),3))

    #hologram = fixhololevel(hologram)
    datahome, basename = os.path.split(sys.argv[imgptr])
    datahome, datalastdir = os.path.split(datahome)

    if fftpad > 0:
        paddedholo = padding.copy()
        paddedholo[fftpad:-fftpad,fftpad:-fftpad] = hologram
    else:
        paddedholo = hologram.copy()

    cimage = makefig(paddedholo.get(), zi, imgptr)
    if type(cimage) is np.ndarray:
        cimadj = np.abs(cimage.copy())
    else:
        cimadj = np.abs(cimage.get())

    cv2.putText(cimadj,f"z={zees[zi]:09.06f} zi={zi:04d} frame={os.path.basename(sys.argv[imgptr])}", (1,51),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,0),2)
    cv2.putText(cimadj,f"z={zees[zi]:09.06f} zi={zi:04d} frame={os.path.basename(sys.argv[imgptr])}", (0,50),cv2.FONT_HERSHEY_DUPLEX,0.4,(255,255,155),1)

    while True:
        if type(cimadj) is not np.ndarray:
            cimadj = cimadj.get()
        cv2.imshow("image", cimadj)
        print(f"{imgptr},{zi}", file=open(f"{datahome}/curpos.csv", "w"))
        print(f"CURPOS:{imgptr},{zi}")
        k=cv2.waitKey(0) & 255
        if k == 27 or k == ord('q'):
            escaped=True
            break

        # VI-like navigation keys
        # h - previous image
        if k == ord('h'):
            if imgptr > 1:
                imgptr -= 1
                break
        # l - next image
        if k == ord('l'):
            if imgptr < len(sys.argv) - 1:
                imgptr += 1
                break
        # j - lower z value
        if k == ord('j'):
            if zi > 0:
                zi -= 1
                break
        # k - higher z value
        if k == ord('k'):
            if zi < len(zees) - 1:
                zi += 1
                break
        # n - lower by 10 z values
        if k == ord('n'):
            if zi > 10:
                zi -= 10
            else:
                zi  = 0
            break
        # i - higher by 10 z values
        if k == ord('i'):
            if zi < len(zees) - 11:
                zi += 10
            else:
                zi = len(zees)-1
            break
        # b - lower by 100 z values
        if k == ord('b'):
            if zi > 100:
                zi -= 100
            else:
                zi  = 0
            break
        # o - higher by 100 z values
        if k == ord('o'):
            if zi < len(zees) - 101:
                zi += 100
            else:
                zi = len(zees)-1
            break

