from __future__ import print_function

import argparse
import grpc

import os
import api_pb2
import api_pb2_grpc

def getFiles(directory):
  if os.path.isfile(directory):
    return [getFileData(directory)]
  elif os.path.isdir(directory):
    return iterateImages(directory)

def iterateImages(directory):
  exts = [".jpg", ".jpeg", ".png"]
  for subdir, dirs, files in os.walk(directory):
    for path in files:
      (imageClass, fName) = (os.path.basename(subdir), path)
      (imageName, ext) = os.path.splitext(fName)
      if ext.lower() in exts:
        yield getFileData(os.path.join(subdir, path))

def getFileData(file):
  in_file = open(file, "rb")
  data = in_file.read()
  in_file.close()
  return data

def reloadModel(args):
  channel = grpc.insecure_channel(args.apiUrl)
  stub = api_pb2_grpc.ApiStub(channel)
  response = stub.Reload(api_pb2.ReloadRequest())

def save(args):
  imgs = list(getFiles(args.inputDir))
  channel = grpc.insecure_channel(args.apiUrl)
  stub = api_pb2_grpc.ApiStub(channel)
  response = stub.AddFaces(api_pb2.AddFacesRequest(name='uncategorized', images=imgs))
  print(response)

def infer(args):
  imgs = list(getFiles(args.inputDir))
  channel = grpc.insecure_channel(args.apiUrl)
  stub = api_pb2_grpc.ApiStub(channel)
  response = stub.Infer(api_pb2.InferRequest(images=imgs, isAligned=args.preAligned))
  for img in response.predictions:
    print("Predict {} with {:.2f} confidence.".format(img.name, img.confidence))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('inputDir', type=str, help="Input image.")
  parser.add_argument('--apiUrl', type=str, help="apisUrl",
                      default='localhost:50051')
  subparsers = parser.add_subparsers(dest='mode', help="Mode")
  computeMeanParser = subparsers.add_parser('save', help='add faces')
  alignmentParser = subparsers.add_parser('infer', help='infer from image')
  alignmentParser.add_argument('--preAligned', action='store_true')
  reloadParser = subparsers.add_parser('reload', help='reload the model')

  args = parser.parse_args()

  if args.mode == 'save':
    save(args)
  elif args.mode == 'infer':
    infer(args)
  elif args.mode == 'reload':
    reloadModel(args)
