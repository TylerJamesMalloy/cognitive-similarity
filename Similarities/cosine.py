"""
MIT License

Copyright (c) 2024 Tyler Malloy, Cleotilde Gonzalez Dynamic Decision Making Labratory, Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math 
import numpy as np 
from .similarity import Similarity

class Cosine(Similarity):
    def __init__(self, name="Human", categories=[], args=None, m=None, w=None, n=False, p=False, weighted=False, pruned=False, **kwargs):
        super().__init__(name, categories, args)
        self.annotations = None 
        self.categories = categories
        self.column = args.typeColumn 
        self.categoryMeans = {}
        
        self.args=args 
        self._m = m 
        self._w = w 
        self._n = n 
        self._p = p

        self.weighted = weighted
        self.pruned = pruned
        
         
    def set_m(self, m):
        self._m = m
    def set_w(self, w):
        self._w = w
    def set_n(self, n):
        self._n = n
    def set_p(self, p):
        self._p = p

    def set_documents(self, documents):
        super().set_documents(documents)
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
        super().set_annotations(annotations)

        # Calculate weights based on the difference between category mean embedding weights. 
        if(self.weighted): 
            self._n = 0.5
            category_embedding_means = []
            for category in self.categories:
                category_embedding = []
                for id in annotations[annotations[self.args.annotationColumn] == category][self.args.idColumn].to_list():
                    if(len(self.documents[self.documents[self.args.idColumn] == id]['Embedding'].to_list()) == 0): continue 
                    embedding = self.documents[self.documents[self.args.idColumn] == id]['Embedding'].to_list()[0]
                    embedding = [float(x) for x in embedding]
                    category_embedding.append(embedding)
                category_embedding = np.array(category_embedding)
                category_embedding = np.mean(category_embedding, axis=0)
                category_embedding_means.append(category_embedding.tolist())
            
            category_embedding_means = np.array(category_embedding_means)
            self._w = (category_embedding_means.var(axis=0) ** 2)
            #self._w = self._w  / np.sum(self._w)
        
        if(self.pruned): 
            self._n = 0.5
            category_embedding_means = []
            for category in self.categories:
                category_embedding = []
                for id in annotations[annotations[self.args.annotationColumn] == category][self.args.idColumn].to_list():
                    if(len(self.documents[self.documents[self.args.idColumn] == id]['Embedding'].to_list()) == 0): continue 
                    embedding = self.documents[self.documents[self.args.idColumn] == id]['Embedding'].to_list()[0]
                    embedding = [float(x) for x in embedding]
                    category_embedding.append(embedding)
                category_embedding = np.array(category_embedding)
                category_embedding = np.mean(category_embedding, axis=0)
                category_embedding_means.append(category_embedding.tolist())
            
            category_embedding_means = np.array(category_embedding_means)
            self._w = category_embedding_means.var(axis=0)
            threshold = np.mean(self._w) # could make this a command line arugment 
            self._w[self._w < threshold] = 0
            self._w = self._w / np.sum(self._w)
    
    def similarity(self, u):
        """
        Calculate the similarity of the document u to each category, or to a specific subset if categories is not None
        """
        categorySimilarities = {}
        for category, mean in zip(self.categoryMeans.keys(), self.categoryMeans.values()):
            if(self._w is not None):
                    w = self._w 
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
            if(self._n is not None):
                sim = float(sim - self._n)
            else:
                sim = float(sim)
                
            categorySimilarities[category] = sim

        return categorySimilarities 