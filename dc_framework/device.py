from functools import wraps

import torch

#I don't have GPU to test it :D

def use_gpu(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        # print(f"method: {method}")
        # print(f"kwargs: {kwargs}")
        # print(f"args[0]: {args[0]}") #self
        # print(f"defaults: {method.__defaults__}")
        
        if kwargs.get('gpu', True) != False and torch.cuda.is_available():  #to acess gpu from defaults use method.__defaults__ - that's a tuple of defaul arguments
            dev = "cuda:0"
        else:
            dev = "cpu"
        print(f"Using {dev}")
        
        args[0].model.to(dev)
        args[0].device = dev
        # print(f"model: {args[0].model} \n on device {args[0].device}")
        return method(*args, **kwargs)
    return wrapper

        
        
        
