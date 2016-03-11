import cv2
import numpy as np
import matplotlib.pyplot as plt

# built-in modules
from multiprocessing.pool import ThreadPool

import cv2

import digits2


# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# Take Red families and plot them; 0=red triangles
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

# Take Blue families and plot them; 1=blue squares
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')


newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = digits2.KNearest()
knn.train(trainData,responses)
ret, results, neighbours ,dist = knn.model.findNearest(newcomer, 3)

print "result: ", results,"\n"
print "neighbours: ", neighbours,"\n"
print "distance: ", dist

# 10 new comers
#newcomers = np.random.randint(0,100,(10,2)).astype(np.float32)
#ret, results,neighbours,dist = knn.model.findNearest(newcomer, 3)
# The results also will contain 10 labels.

plt.show()