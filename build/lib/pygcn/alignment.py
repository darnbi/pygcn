import time
import argparse
import numpy as np

# calculate alignment matrix with dot product which computes similarity
def naive_alignment(embed_1, embed_2):

    node_num1 = embed_1.shape[0]
    node_num2 = embed_2.shape[0]
    i = 0
    j = 0
    
    #rank = [[0] * node_num1] * node_num2
    rank = np.zeros((node_num1, node_num2))
    
    for row1 in embed_1:
        j = 0
        for row2 in embed_2:
            
            #get the similarity between current node in embed_1 and all the node in embed_2
            rank[i, j] = np.dot(row1, row2)
            j = j+1
        i = i+1
    #return final alignment matrix       
    return rank

def alignment(embed_1, embed_2):
    S = np.matmul(embed_1, embed_2.transpose())
    return S
