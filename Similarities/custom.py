import math 
import numpy as np 
from .similarity import Similarity

class Custom(Similarity):
    def __init__(self, name="Custom", categories=[], args=None, **kwargs):
        super().__init__(name, categories, args)
    
    """
    This similarity metric assumes that the annotation dataframe contains information 
    on the annotation confidence and reaction time. If these are not present then 
    this similarity will default to include whatever information is available. 
    """
    def similarity(self, u, categories=None):
        return {'phishing':0, 'ham':1}

"""
Edit the below code and the object to the __init__.py file to add your own similarity metric
"""
class MyOwnSimilarityMetric(Similarity):
    def __init__(self, name="Custom", categories=[], args=None, **kwargs):
        super().__init__(name, categories, args)
    
    """
    This similarity metric assumes that the annotation dataframe contains information 
    on the annotation confidence and reaction time. If these are not present then 
    this similarity will default to include whatever information is available. 
    """
    def similarity(self, u, categories=None):
        return {'phishing':0, 'ham':1}
                