# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import api_pb2 as api__pb2


class ApiStub(object):
  """The greeting service definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Infer = channel.unary_unary(
        '/api.Api/Infer',
        request_serializer=api__pb2.InferRequest.SerializeToString,
        response_deserializer=api__pb2.InferResponse.FromString,
        )
    self.AddFaces = channel.unary_unary(
        '/api.Api/AddFaces',
        request_serializer=api__pb2.AddFacesRequest.SerializeToString,
        response_deserializer=api__pb2.AddFacesResponse.FromString,
        )
    self.Reload = channel.unary_unary(
        '/api.Api/Reload',
        request_serializer=api__pb2.ReloadRequest.SerializeToString,
        response_deserializer=api__pb2.ReloadResponse.FromString,
        )


class ApiServicer(object):
  """The greeting service definition.
  """

  def Infer(self, request, context):
    """Sends a greeting
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def AddFaces(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Reload(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ApiServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Infer': grpc.unary_unary_rpc_method_handler(
          servicer.Infer,
          request_deserializer=api__pb2.InferRequest.FromString,
          response_serializer=api__pb2.InferResponse.SerializeToString,
      ),
      'AddFaces': grpc.unary_unary_rpc_method_handler(
          servicer.AddFaces,
          request_deserializer=api__pb2.AddFacesRequest.FromString,
          response_serializer=api__pb2.AddFacesResponse.SerializeToString,
      ),
      'Reload': grpc.unary_unary_rpc_method_handler(
          servicer.Reload,
          request_deserializer=api__pb2.ReloadRequest.FromString,
          response_serializer=api__pb2.ReloadResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'api.Api', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
