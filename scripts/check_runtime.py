import importlib.util

print('torch spec:', importlib.util.find_spec('torch'))
print('facenet_pytorch spec:', importlib.util.find_spec('facenet_pytorch'))

try:
    import torch
    print('torch import ok; version:', getattr(torch, '__version__', '<no __version__>'))
    print('cuda available:', torch.cuda.is_available())
except Exception as e:
    print('torch import error:', repr(e))

try:
    import facenet_pytorch
    print('facenet_pytorch import ok; module:', getattr(facenet_pytorch, '__version__', facenet_pytorch))
except Exception as e:
    print('facenet_pytorch import error:', repr(e))
