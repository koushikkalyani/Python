import numpy as np
from itertools import islice
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray

def rgb2gray(rgb):  # This def is used to convert RGB image to Gray image..
    return (np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])).astype(np.uint8)

def matrixmultsum(A,B,m,n):  # This def is used to multiply corresponding elements of two equal size matrices and return scalar value..
  sum = 0
  for i in range(m):            # runs over rows
    for j in range(n):          # runs over columns
      sum += A[i][j]*B[i][j]    # sum all results and return scalr
  return sum

# def patch(A,m,n,i,j):
#  if (i+m) <= A.shape[0]:
#     if(j+n) <= A.shape[1]:
#      result = [list(islice(row, j, j+n)) for row in islice(A, i, i+m)] 
#  return result
def patch(A,m,n,i,j):  # For even sized kernel.. 
 if (i+m) <= A.shape[0]:
    if(j+n) <= A.shape[1]:
     result = A[i:i+m,j:j+n] 
 return result

def patch2(A,m,n,i,j):            # Takes image and returns just patch of it with size (m,n) and the index (i,j) 
  return A[i-m:i+m+1,j-n:j+n+1]   # This is for odd sized kernels..

def Convolution(im,kernel):                                             # This def performs convolution, given a image and kernel...
  if (kernel.shape[0] % 2 == 0) and (kernel.shape[1] % 2 == 0) :        # For even sized kernels .... 
    iterationr = im.shape[0] - kernel.shape[0] + 1                      # no of rows for output image ...
    iterationc = im.shape[1] - kernel.shape[1] + 1                      # no of columns for output image ...
    out = np.zeros((iterationr,iterationc),dtype=np.uint8)
    for i in range(iterationr):
      for j in range(iterationc):
        pat = patch(im,kernel.shape[0],kernel.shape[1],i,j)
        out[i,j] = matrixmultsum(pat,kernel,kernel.shape[0],kernel.shape[1]) 

  else:   # For odd sized kernels .... 
    iteration = im.shape[0] * im.shape[1]
    padding_size = (kernel.shape[0] - 1)//2                         # Padding size to create output image....
    padded_im = np.zeros((im.shape[0] + (2 * padding_size), im.shape[1] + (2 * padding_size)),dtype=np.uint8)
    out = np.zeros((im.shape[0],im.shape[1]),dtype=np.uint8)
    for i in range(padding_size,padding_size + im.shape[0]):
      for j in range(padding_size,padding_size + im.shape[1]):
        padded_im[i,j] = im[i-padding_size,j-padding_size]          # Padded image created, ready for convulation...
    for i in range(padding_size,padding_size + im.shape[0]):
      for j in range(padding_size,padding_size + im.shape[1]):
        pat = patch2(padded_im,padding_size,padding_size,i,j)
        out[i-padding_size,j-padding_size] = matrixmultsum(pat,kernel,kernel.shape[0],kernel.shape[1]) 
  return out  

# Read image from your storage, replace "\" with "/" in copied address of image..
img = ((mpimg.imread("D:/KOUSHIK'S WORK/TM/GraphTsetlinMachine-master/GraphTsetlinMachine-master/GraphTsetlinMachine/kenshin.png")) *255 ).astype(np.uint8)  
gray = rgb2gray(img)*255  # Convert rgb to gray.
filter =np.array([[-1,0,10],[-1,0,10],[-1,0,10]]) # Filter or kernel, change as per application..
filtered_output= Convolution(gray,filter)  # Call for convolution..
plt.imshow(filtered_output, cmap=plt.get_cmap('gray'), vmin=0, vmax=255) # Plot output for visualization..
plt.show()

