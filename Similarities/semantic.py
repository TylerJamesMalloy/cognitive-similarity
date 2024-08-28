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

class Semantic(Similarity):
    def __init__(self, name="Human", categories=[], args=None, **kwargs):
        super().__init__(name, categories, args)
    
    """
    This similarity metric assumes that the annotation dataframe contains information 
    on the annotation confidence and reaction time. If these are not present then 
    this similarity will default to include whatever information is available. 
    """
    def set_annotations(self, annotations):
        super().set_annotations(annotations)

        # Calculate weights based on the difference between category mean embedding weights.  
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

    def similarity(self, u):
        doc = self.documents[self.documents["Embedding"] == tuple(u)].iloc[0]
        docId = doc[self.args.idColumn].item()

        assert(False)
        docAnnotations = self.participant[self.participant[self.args.idColumn] == docId]
        
        for idx, docAnnotation in docAnnotations.iterrows():
            similarity = 1
            if('Confidence' in docAnnotations.columns):
                similarity *= docAnnotation['Confidence'] / 5
            if('ReactionTime' in docAnnotations.columns):
                similarity *= np.abs((docAnnotation['ReactionTime'] - self.participant['ReactionTime'].max()) /  self.participant['ReactionTime'].max())

            if(docAnnotation['Decision'] == 'false'): # make this more general 
                return {'ham':similarity, 'phishing':1-similarity}
            else:
                return {'phishing':similarity, 'ham':1-similarity}