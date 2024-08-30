"""
MIT License

Copyright (c) (Removed for anonymous submission)

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
from .cosine import Cosine

class Semantic(Similarity):
    def __init__(self, name="Semantic", categories=[], args=None, **kwargs):
        super().__init__(name, categories, args)
        self.semanticCategoryMeans = {}
        self.cosine = Cosine(name="Cosine", categories=categories, args=args, weighted=True, kwargs=kwargs)
    
    def set_documents(self, documents):
        self.cosine.set_documents(documents)
        return super().set_documents(documents)
    
    def set_annotations(self, annotations):
        super().set_annotations(annotations)
        self.cosine.set_annotations(annotations)
        # Calculate weights based on the difference between category mean embedding weights.  
        self._n = 0.5
        for category in self.categories:
            semanticCategoryMean = []
            for id in annotations[annotations[self.args.annotationColumn] == category][self.args.idColumn].to_list():
                doc = self.documents[self.documents[self.args.idColumn] == id]
                semanticCategories = []
                for semanticCat in self.args.semanticCategories.split(","):
                    if(len(doc[semanticCat]) != 1): 
                        semanticCategories = None 
                        break  
                    semanticCategories.append(doc[semanticCat].item())
                if(semanticCategories is not None):
                    semanticCategoryMean.append(semanticCategories)

            semanticCategoryMean = np.array(semanticCategoryMean)
            semanticCategoryMean = np.mean(semanticCategoryMean, axis=0)
            semanticCategoryMean = np.exp(semanticCategoryMean/0.1) / np.sum(np.exp(semanticCategoryMean/0.1))
            self.semanticCategoryMeans[category] = np.array(semanticCategoryMean)
        

    def similarity(self, u):
        similarities = {}
        doc = self.documents[self.documents["Embedding"] == tuple(u)].iloc[0]
        semanticCategories = []
        for semanticCat in self.args.semanticCategories.split(","):
            semanticCategories.append(doc[semanticCat])

        for category, mean in zip(self.semanticCategoryMeans.keys(), self.semanticCategoryMeans.values()):
            u = np.array([semanticCategory if semanticCategory > 0 else 1e-1 for semanticCategory in semanticCategories])
            #u = np.exp(u/0.1) / np.sum(np.exp(u/0.1))
            v = np.array(mean)
            uv = np.dot(u, v)
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            sim = 1.0 - uv / math.sqrt(uu * vv)
            similarities[category] = (float(sim))

        return similarities