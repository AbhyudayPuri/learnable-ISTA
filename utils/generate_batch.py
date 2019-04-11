from sklearn.feature_extraction import image
import cv2 as cv
import glob
from random import shuffle


path = '../data/train/*.jpg'
files = glob.glob(path)

for file in files:
	im = cv.imread(file)

fp = open('../data/iids_train.txt')
lines = fp.read().splitlines() # Create a list containing allt lines
fp.close()

im = cv.imread('../data/train/' + str(lines[0]) + '.jpg')
cv.imshow("image", im)
cv.waitKey(0)
print(lines)
print(len(lines))

shuffle(lines)
print(lines)
