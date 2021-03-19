import numpy as np
import nnfs

nnfs.init()

def SoftMax(inList, dbFlag = False):
    exp_values = np.exp(inList - np.max(inList, axis=1, keepdims=True)) # exponentiation while subtracting from the largest in each row to protect from overflow 
    norm_values = exp_values/ np.sum(exp_values, axis=1, keepdims=True) # normalization
    if dbFlag : #print values if debug flag is set
        print(exp_values)
        print( norm_values)
        print(sum(norm_values))
    return norm_values


