from sklearn.feature_extraction import image
import cv2 as cv
import glob

## Ppath of the folder that contains the data 
path = '../data/train/*.jpg'

## Reading all the files in the directory specified by the path
files = glob.glob(path)   

## Reading the images in the given directory
for file in files:
	im = cv.imread(file)
	## Generate patches for each of the image and store them
	patches = image.extract_patches_2d(im, (10, 10))
	[num_patches, _, _, _] = patches.shape


print(num_patches)
# cv.imshow("image", patches[0])
# cv.waitKey(0)
