from concurrent import futures
import time

import grpc

import api_pb2
import api_pb2_grpc

import uuid
import argparse
import cv2
import os
import pickle
import sys
from subprocess import call

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', '..', 'models')
classifyEx = os.path.join(fileDir, '..', 'classifier.py')
batchRep = os.path.join(fileDir, '..', '..', 'batch-represent', 'main.lua')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Api(api_pb2_grpc.ApiServicer):

  def __init__(self):
    self.loadPickel(args.classifierModel)

  def loadPickel(self, model):
    with open(model, 'rb') as f:
      if sys.version_info[0] < 3:
        (le, clf) = pickle.load(f)
      else:
        (le, clf) = pickle.load(f, encoding='latin1')
    self.le = le
    self.clf = clf

  def AddFaces(self, request, context):
    results = []
    directory = os.path.join(args.facesDirectory, request.name)
    if not os.path.isdir(directory):
      os.makedirs(directory)
    for img in request.images:
      ids = []
      reps = self.getRep(img)
      for r in reps:
        bgrImg = cv2.cvtColor(r[2], cv2.COLOR_RGB2BGR)
        id = str(uuid.uuid4())
        cv2.imwrite(os.path.join(directory, id + ".jpg"), bgrImg)
        ids.append(id)
      results.append(api_pb2.FoundFace(faceCount=len(reps), imageIds=ids))

    return api_pb2.AddFacesResponse(found=results)

  def Reload(self, request, context):
    try:
      os.remove(os.path.join(args.facesDirectory, 'cache.t7'))
    except OSError:
      pass
    call([batchRep, '-outDir', args.featureDirectory, '-data', args.facesDirectory])
    self.train()
    self.loadPickel(args.classifierModel)
    return api_pb2.ReloadResponse()

  def Infer(self, request, context):
    results = []
    for img in request.images:
      reps = self.getRep(img, request.isAligned)
      for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
        predictions = self.clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = self.le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if isinstance(self.clf, GMM):
          dist = np.linalg.norm(rep - self.clf.means_[maxI])
          print("  + Distance from the mean: {}".format(dist))
        results.append(api_pb2.Prediction(name=person.decode('utf-8'), confidence=confidence))

    return api_pb2.InferResponse(predictions=results)

  def loadImageFromBytes(self, image):
    nparr = np.fromstring(image, np.uint8)
    bgrImg = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
    if bgrImg is None:
      raise Exception("Unable to load image")
    return bgrImg

  def train(self):
      print("Loading embeddings.")
      fname = "{}/labels.csv".format(args.featureDirectory)
      labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
      labels = map(itemgetter(1),
                   map(os.path.split,
                       map(os.path.dirname, labels)))  # Get the directory.
      fname = "{}/reps.csv".format(args.featureDirectory)
      embeddings = pd.read_csv(fname, header=None).as_matrix()
      le = LabelEncoder().fit(labels)
      labelsNum = le.transform(labels)
      nClasses = len(le.classes_)
      print("Training for {} classes.".format(nClasses))

      if args.classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)
      elif args.classifier == 'GridSearchSvm':
        print("""
        Warning: In our experiences, using a grid search over SVM hyper-parameters only
        gives marginally better performance than a linear SVM with C=1 and
        is not worth the extra computations of performing a grid search.
        """)
        param_grid = [
            {'C': [1, 10, 100, 1000],
              'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
              'gamma': [0.001, 0.0001],
              'kernel': ['rbf']}
        ]
        clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
      elif args.classifier == 'GMM':  # Doesn't work best
        clf = GMM(n_components=nClasses)

      # ref:
      # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
      elif args.classifier == 'RadialSvm':  # Radial Basis Function kernel
        # works better with C = 1 and gamma = 2
        clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
      elif args.classifier == 'DecisionTree':  # Doesn't work best
        clf = DecisionTreeClassifier(max_depth=20)
      elif args.classifier == 'GaussianNB':
        clf = GaussianNB()

      # ref: https://jessesw.com/Deep-Learning/
      elif args.classifier == 'DBN':
        from nolearn.dbn import DBN
        clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                  learn_rates=0.3,
                  # Smaller steps mean a possibly more accurate result, but the
                  # training will take longer
                  learn_rate_decays=0.9,
                  # a factor the initial learning rate will be multiplied by
                  # after each iteration of the training
                  epochs=300,  # no of iternation
                  # dropouts = 0.25, # Express the percentage of nodes that
                  # will be randomly dropped as a decimal.
                  verbose=1)

      if args.ldaDim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                        ('clf', clf_final)])

      clf.fit(embeddings, labelsNum)

      fName = "{}/classifier.pkl".format(args.featureDirectory)
      print("Saving classifier to '{}'".format(fName))
      with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

  def getRep(self, imgReq, isAligned=False):
    bgrImg = self.loadImageFromBytes(imgReq)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bbs = align.getAllFaceBoundingBoxes(rgbImg)

    reps = []
    for bb in bbs:
      if isAligned:
        alignedFace = rgbImg
      else:
        alignedFace = align.align(
          args.imgDim,
          rgbImg,
          bb,
          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
          raise Exception("Unable to align image: {}".format(imgPath))

      rep = net.forward(alignedFace)
      reps.append((bb.center().x, rep, alignedFace))

    return sorted(reps, key=lambda x: x[0])

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  api_pb2_grpc.add_ApiServicer_to_server(Api(), server)
  server.add_insecure_port('[::]:50051')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ldaDim', type=int, default=-1)
    parser.add_argument(
        '--classifier',
        type=str,
        choices=[
            'LinearSvm',
            'GridSearchSvm',
            'GMM',
            'RadialSvm',
            'DecisionTree',
            'GaussianNB',
            'DBN'],
        help='The type of classifier to use.',
        default='LinearSvm')
    parser.add_argument(
        '--featureDirectory',
        type=str,
        help="path to feature directory")

    parser.add_argument(
        '--facesDirectory',
        type=str,
        help="path to faces directory")

    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))

    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))

    parser.add_argument(
        '--classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(
            time.time() - start))

    start = time.time()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)
    serve()
