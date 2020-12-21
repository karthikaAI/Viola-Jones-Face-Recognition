import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
from dask import delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
import os
import glob
from PIL import Image
from sklearn import metrics

os.getcwd()
os.chdir('D:\CVproject02')
def in_array():
    Fimg = glob.glob("D:\\CVproject02\\Face16\\*.bmp")
    NFimg = glob.glob("D:\\CVproject02\\Nonface16\\*.bmp")
    p=len(Fimg[1600:2700])
    q=len(NFimg[2700:4800])
    data_x=Fimg[1600:2700]+ NFimg[2700:4800]    
    x = np.array([np.array(Image.open(fname)) for fname in data_x])
    return x,p,q

def test_array():
    Fimg = glob.glob("D:\\CVproject02\\Face16\\*.bmp")
    NFimg = glob.glob("D:\\CVproject02\\Nonface16\\*.bmp")
    p=len(Fimg[1600:1700])
    q=len(NFimg[2700:2800])
    data_x=Fimg[1600:1700]+ NFimg[2700:2800]    
    x = np.array([np.array(Image.open(fname)) for fname in data_x])
    return x,p,q

def extract_feature_image(img, feature_type, feature_coord):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)
images = in_array()[0]
face_count=in_array()[1]
non_face_count=in_array()[2]

feature_types = ['type-2-x','type-3-x','type-3-y','type-4','type-2-y']

# Build a computation graph using Dask. This allows the use of multiple
# CPU cores later during the actual computation
X = delayed(extract_feature_image(img, feature_types, None) for img in images)
t_start = time()
X = np.array(X.compute(scheduler='threads'))
N=X.shape[0]
time_full_feature_comp = time() - t_start
y = np.array([1] * face_count + [-1] * non_face_count)
feature_coord, feature_type = \
    haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
                            feature_type=feature_types)
#Threshold Computation
haar_face=X[:100,:]
haar_non_face=X[100:200,:]
mean_face=np.mean(haar_face,axis=0)
mean_non_face=np.mean(haar_non_face,axis=0)
temp=np.stack((mean_face,mean_non_face))
thresh=np.mean(temp,axis=0)


t=10
a=[]
h_x_=[]
h_t_x_=[]
weights=np.ones(N)/N
weak_classifiers=[]
for t in range(t):
#Adaboost   
    e_h = np.empty((thresh.shape))
    k=X>=thresh
    h_x_not_y = np.vstack((np.where(k[:face_count,:]==False,1,0*k[:face_count,:]),\
                               np.where(~k[face_count:,:]==False,1,0*~k[face_count:,:])))
    e_h=np.dot(weights,h_x_not_y)
    min=np.argsort(e_h)[0]
    weak_classifiers=np.append(weak_classifiers,min)
    min_e=e_h[min]
    alpha=0.5*np.log((1-min_e)/min_e)
    h_x = np.vstack((np.where(k[:face_count,:]==False,-1,1*k[:face_count,:]),\
                               np.where(~k[face_count:,:]==False,1,-1*~k[face_count:,:])))
    h_t_x=h_x[:,min]
    
    #h_t_x_=np.append(h_t_x_,h_t_x)
    term= -1*alpha*np.multiply(h_t_x,y)
    z_t=np.multiply(weights,np.exp(term)).sum()
    weights=np.multiply(weights,np.exp(term))/z_t
    a.append(alpha)

#Testing
weak_classifiers=weak_classifiers.astype('int')
images_test = test_array()[0]
face_count_test = test_array()[1]
non_face_count_test = test_array()[2]
time_full_feature_comp = time() - t_start
y_test = np.array([1] * face_count_test + [-1] * non_face_count_test)


x_test = delayed(extract_feature_image(img, feature_type[weak_classifiers],feature_coord[weak_classifiers]) for img in images_test)
t_start = time()
x_test = np.array(x_test.compute(scheduler='threads'))


i=np.multiply(x_test,a)
output=np.sign(np.sum(i,axis=1))
s =  output == y_test
err=0
for i in range(s.shape[0]):
    if s[i]==True:
        err=err+1
accuracy =err/images_test.shape[0]
print("accuracy: "+"{:.2%}".format(accuracy))

fig, axes = plt.subplots(5, 2)
for idx, ax in enumerate(axes.ravel()):
    image = images_test[0]
    image = draw_haar_like_feature(image, 0, 0,16,16,
                                   [feature_coord[weak_classifiers[idx]]])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([]) 

_ = fig.suptitle('The most important features after boosting')

plt.show()
fpr,tpr,h=metrics.roc_curve(y_test,s)
plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
