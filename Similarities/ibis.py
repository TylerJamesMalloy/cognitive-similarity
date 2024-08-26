import math 
import numpy as np 
from .similarity import Similarity

class IBIS(Similarity):
    def __init__(self, name="IBIS", categories=[], args=None, **kwargs):
        super().__init__(name, categories, args)
    
    """
    This similarity metric assumes that the annotation dataframe contains information 
    on the annotation confidence and reaction time. If these are not present then 
    this similarity will default to include whatever information is available. 
    """
    def similarity(self, u, categories=None):
        return {'phishing':0, 'ham':1}
                