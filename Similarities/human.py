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

class Human(Similarity):
    def __init__(self, name="Human", categories=[], args=None, **kwargs):
        super().__init__(name, categories, args)
    
    """
    This similarity metric assumes that the annotation dataframe contains information 
    on the annotation confidence and reaction time. If these are not present then 
    this similarity will default to include whatever information is available. 
    """
    def similarity(self, u):
        if(len(self.documents[self.documents["Embedding"] == tuple(u)]) == 0): return None 
        doc = self.documents[self.documents["Embedding"] == tuple(u)].iloc[0]
        docId = doc[self.args.idColumn].item()
        docAnnotations = self.annotations[self.annotations[self.args.idColumn] == docId]
        
        similarities = {}
        for category in self.categories:
            catAnnotations = docAnnotations[docAnnotations[self.args.annotationColumn] == category]
            if(len(catAnnotations) == 0):
                similarities[category] = None
                continue 

            similarity = len(catAnnotations) / len(docAnnotations)
            if('Confidence' in catAnnotations.columns):
                similarity *= np.abs(catAnnotations['Confidence'].mean() / self.annotations['Confidence'].max())
            if('ReactionTime' in catAnnotations.columns):
                similarity *= np.abs((catAnnotations['ReactionTime'].mean() - self.annotations['ReactionTime'].max()) /  self.annotations['ReactionTime'].max())
            
            similarities[category] = float(similarity)

        if(len(self.categories) == 2): # If there are exactly 2 categories we can replace no value similarities with the inverse of the other category similarity. 
            for category_index, category in enumerate(self.categories):
                if(similarities[category] is None):
                    other_category = 1 if category_index == 0 else 0
                    if(similarities[self.categories[other_category]] is None): 
                        return None # if both are none we failed to calculate similarity 
                    similarities[category] = float(0.75 - np.clip(similarities[self.categories[other_category]], a_min=0, a_max=0.75) )

        return similarities