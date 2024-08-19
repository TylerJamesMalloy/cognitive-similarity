import math 
import numpy as np 
from .similarity import Similarity

class Cosine(Similarity):
    def __init__(self, name="Similarity"):
        super().__init__(name)
    
    def similarity(self, u, v, m=None, w=None, norm=True):
        if(w is not None):
            if(norm):
                w = w / np.sum(w)
                vw = v * w
                uw = u * w
            else:
                vw = v
                uw = u 
            uv = np.dot(u, vw)
            uu = np.dot(u, uw)
            vv = np.dot(v, vw)
            dist = 1.0 - uv / math.sqrt(uu * vv)
            sim = 1 - (np.clip(dist, 0.0, 2.0)  / 2)
            if(m is not None):
                if(sim < m):
                    sim = 1
                else:
                    sim = 0
        return sim 
        
        return 