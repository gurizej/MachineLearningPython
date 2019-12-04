import mahotas as mh
from matplotlib import pyplot as plt
import numpy as np


imageDIR = "../SimpleImageDataset"

image = mh.imread('../SimpleImageDataset/scene00.jpg')

plt.imshow(image)


#Display image in a greyscale
image = mh.colors.rgb2grey(image, dtype=np.uint8)
plt.imshow(image)
plt.gray()

#Threshholding, essentially just turning it into solid black or solid white image
#this helps reduce the noise in the picture
thresh = mh.thresholding.otsu(image)
print('Otsu threshold is {}.'.format(thresh))
plt.imshow(image > thresh)

plt.show()