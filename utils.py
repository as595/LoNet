from PIL import Image
import numpy as np
import pylab as pl
from scipy import signal


class LoData():

    def __init__(self, filepath):

        self.filepath = filepath

    def read_png(self):

        self.open_png()
        self.mask_png()
        self.find_nbeams()
        self.get_corners()
        self.extract_data()

        return


    def open_png(self):

        im = Image.open(self.filepath)
        im = np.array(im)
        im = im[:,:,0:3]

        self.im = im

        return


    def mask_png(self):

        ref = self.im[:,:,0]
        ref = np.tile(ref[:, :, np.newaxis], (1,1,3))

        mask = np.zeros([self.im.shape[0],self.im.shape[1]])
        for i in range(0,self.im.shape[0]):
            for j in range(0,self.im.shape[1]):
                if np.all(self.im[i,j,:]==ref[i,j,:]): mask[i,j] = 1
                
        i = 0
        while True:
            if not np.all(mask[i,:]==mask[i,0]): 
                i0 = i
                break
            i+=1
        i = mask.shape[0]-1
        while True:
            if not np.all(mask[i,:]==mask[i,0]): 
                i1 = i
                break
            i-=1

        j = 0
        while True:
            if not np.all(mask[:,j]==mask[0,j]): 
                j0 = j
                break
            j+=1
        j = mask.shape[1]-1
        while True:
            if not np.all(mask[:,j]==mask[0,j]): 
                j1 = j
                break
            j-=1

        self.mask = mask

        return


    def find_nbeams(self):

        frac = np.count_nonzero(self.mask)/(self.mask.shape[0]*self.mask.shape[1])
        if frac<0.40: 
            nbeams=1
        elif frac<0.46:
            nbeams=2
        else:
            nbeams=4

        self.nbeams = nbeams

        return

    def get_corners(self):

        if self.nbeams==1:
            self.corners = [[[100,60],[618,478]]]
        if self.nbeams==2:
            self.corners = [[[100,60],[618,216]],[[100,322],[618,216]]]
        if self.nbeams==4:
            self.corners = [[[100,60],[618,102]],[[100,185],[618,102]],[[100,310],[618,102]],[[100,436],[618,102]]]

        return

    def extract_data(self):

        self.data = np.zeros((self.nbeams,102,618,3), dtype=int)
        for i in range(0,self.nbeams):
            x0 = self.corners[i][0][0]
            y0 = self.corners[i][0][1]
            dx = self.corners[i][1][0]
            dy = self.corners[i][1][1]

            data = self.im[y0:y0+dy,x0:x0+dx,:]

            if self.nbeams==4:
                self.data[i,:,:,:] = data
            elif self.nbeams!=4:
                downsampled = signal.resample(data, 102, axis=0)
                self.data[i,:,:,:] = downsampled.astype(int)
            
        return

    