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
    def similarity(self, u, categories=None):
        doc = self.documents[self.documents["Embedding"] == tuple(u)].iloc[0]
        docId = doc[self.args.idColumn].item()
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