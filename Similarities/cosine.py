import math 
import numpy as np 
from .similarity import Similarity

class Cosine(Similarity):
    def __init__(self, name="Similarity", m=None, w=None, n=False, p=False, args=None, **kwargs):
        super().__init__(name)
        self.annotations = None 
        self.categories = None
        self.column = None 
        self.categoryMeans = {}
        
        self.args=args 
        self._m = m 
        self._w = w 
        self._n = n 
        self._p = p
        
         
    def set_m(self, m):
        self._m = m
    def set_w(self, w):
        self._w = w
    def set_n(self, n):
        self._n = n
    def set_p(self, p):
        self._p = p
    
    def set_categories(self, categories, column):
        self.categories = categories
        self.column     = column

    def set_documents(self, documents):
        self.documents = documents
        if(self.categories is None):
            raise Exception("Document categories must be set before setting the documents.")  
        else:
            for category in self.categories:
                categoryAnnotations = self.documents[self.documents[self.column] == category]
                means = categoryAnnotations['Embedding'].to_list()
                if(isinstance(means[0], tuple)):
                    means = np.array([mean for mean in means])
                if(isinstance(means, np.ndarray)):
                    means = np.mean(means, axis=0) # could make the axis an args option but it should be 0 typically 
                self.categoryMeans[category] = means
    
    def set_annotations(self, annotations):
        self.annotations = annotations
    
    def similarity(self, u, categories=None):
        """
        Calculate the similarity of the document u to each category, or to a specific subset if categories is not None
        """
        categorySimilarities = {}
        for category, mean in zip(self.categoryMeans.keys(), self.categoryMeans.values()):
            if(self._w is not None):
                if(self._n):
                    w = self._w / np.sum(w)
                    vw = mean * w
                    uw = u * w
            else:
                vw = mean
                uw = u 
                
            uv = np.dot(u, vw)
            uu = np.dot(u, uw)
            vv = np.dot(mean, vw)
            dist = 1.0 - uv / math.sqrt(uu * vv)
            sim = 1 - (np.clip(dist, 0.0, 2.0)  / 2)
            if(self._m is not None):
                if(sim < self._m):
                    sim = 1
                else:
                    sim = 0
            sim = float(sim)

            categorySimilarities[category] = sim

        return categorySimilarities 


"""
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
"""
        
    