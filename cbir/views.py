from django.shortcuts import render
import joblib
import cv2
import numpy as np
from extract_patches.core import extract_patches
from sklearn import preprocessing
from skimage.feature import local_binary_pattern
from scipy.cluster.vq import *
from django.http import HttpResponse
import json
from django.views import View
from django.views.decorators.csrf import csrf_exempt
import base64

PATCH_SIZE = 15 #Vùng xung quanh key-point
MRSIZE = 6.0 #Đọc thêm

RADIUS = 1 #Bán kính đường tròn, nằm trong vùng -> được tính
NUM_POINTS = 8*RADIUS #Số pixel láng giềng được sử dụng
METHOD = 'uniform' #uniform -> Chỉ giữ lại những giá trị uniform

def extractSIFT(img):
  detector = cv2.xfeatures2d.SIFT_create()
  return detector.detectAndCompute(img, None)

def extractLBP(patches):
  descriptorsLBP = []
  for i, patch in enumerate(patches):
    # print(type(patch))
    descriptor_LBP = local_binary_pattern(patch, NUM_POINTS, RADIUS, METHOD).flatten()
    # print("descriptor %d: %d" % (i, shape(descriptor_LBP)[0]))
    descriptorsLBP.append(descriptor_LBP)
  return descriptorsLBP


def extractFeatures(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kpts, descriptorsSIFT = extractSIFT(img)

    patches = extract_patches(kpts, img, PATCH_SIZE, MRSIZE, 'cv2')
    descriptorsLBP = extractLBP(patches)

    final_descriptor = []
    # print(shape(descriptorsSIFT))

    for desc_tuple in zip(descriptorsSIFT, descriptorsLBP):
        final_descriptor.append([*desc_tuple[0], *desc_tuple[1]])

    return np.array(final_descriptor)
    # return descriptorsSIFT

def query(im):
    # Load the classifier, class names, scaler, number of clusters and vocabulary
    im_features, image_paths, idf, numWords, voc = joblib.load("media/bof_N_SIFT_LBP_8.pkl")

    # List where all the descriptors are stored
    des_list = []

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    des = extractFeatures(im)

    des_list.append(('current', des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]

    #
    test_features = np.zeros((1, numWords), "float32")
    words, distance = vq(descriptors, voc)
    for w in words:
        test_features[0][w] += 1

    # Perform Tf-Idf vectorization and L2 normalization
    test_features = test_features * idf
    test_features = preprocessing.normalize(test_features, norm='l2')

    # Cosine similarity
    score = np.dot(test_features, im_features.T)
    rank_ID = np.argsort(-score)
    # Visualize the results
    res = []

    for i, ID in enumerate(rank_ID[0][0:10]):
        image_full_path = image_paths[ID]
        image_path_split = image_full_path.split('/')
        res.append(image_path_split[len(image_path_split)-1])

    print(len(res))
    return res

def getImage(image_path):
    im = cv2.imread(image_path)
    return im

class QueryView(View):

    def get(self, request, image):
        im = getImage('media/test/' + image)
        results = query(im)
        result_object = {
            "images": results
        }
        return HttpResponse(json.dumps(result_object))

@csrf_exempt
def queryInput(request):
    image = request.POST['image']
    encoded_data = image.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = query(img)
    result_object = {
        "images": results
    }
    return HttpResponse(json.dumps(result_object))