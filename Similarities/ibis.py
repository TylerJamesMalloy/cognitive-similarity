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
from pyibl import Agent 
import copy 
from collections import defaultdict

def sim(u, v, m=None, w=None, norm=True):
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

def get_choices(agent, document):
    base_choice = {}
    for attribute in agent.attributes:
        if(attribute == 'Decision'): continue 
        base_choice[attribute] = document[attribute]
    choices = [copy.deepcopy(base_choice), copy.deepcopy(base_choice)]
    choices[0]['Decision'] = "True"
    choices[1]['Decision'] = "False"

    return choices

class IBIS(Similarity):
    def __init__(self, name="IBIS", categories=[], args=None, **kwargs):
        super().__init__(name, categories, args)
        self.similarities = defaultdict(list)

        
    
    def set_documents(self, documents):
        super().set_documents(documents)

        #self.iblStudent = Agent(name="Embeddings", attributes=["Embedding", "Decision"], mismatch_penalty=20)
        #self.iblStudent.similarity(["Embedding"], lambda x, y: sim(x,y))
        self.iblStudent = Agent(name="Embeddings", attributes=['Sender', 'Subject', 'Sender Mismatch', 'Request Credentials', 'Subject Suspicious', 'Urgent', 'Offer', 'Link Mismatch', "Decision"], mismatch_penalty=20)
        self.iblStudent.similarity(['Sender', 'Subject', 'Sender Mismatch', 'Request Credentials', 'Subject Suspicious', 'Urgent', 'Offer', 'Link Mismatch', "Decision"], lambda x, y: 1 if x == y else 0)
        for index, email in documents.sample(n=20).iterrows():
        #for index, email in documents.iterrows():
            choices = get_choices(self.iblStudent, email)
            if(email["Type"] == "phishing"):
                self.iblStudent.populate([choices[0]], 0.8)
                self.iblStudent.populate([choices[1]], 0.2)
            else:
                self.iblStudent.populate([choices[0]], 0.2)
                self.iblStudent.populate([choices[1]], 0.8)

        self.baseIblStudent = copy.deepcopy(self.iblStudent)

         
    
    def set_annotations(self, annotations):
        super().set_annotations(annotations)
        for userId in annotations["UserId"].unique():
        #for userId in annotations["UserId"].sample(n=1).unique():
            self.iblStudent = copy.deepcopy(self.baseIblStudent) 
            userAnnotations = annotations[annotations["UserId"] == userId]
            for idx, userAnnotation in userAnnotations[::-1].iterrows():
                if(len(self.documents[self.documents["EmailId"] == userAnnotation["EmailId"]]) != 1): continue 
                doc = self.documents[self.documents["EmailId"] == userAnnotation["EmailId"]].iloc[0]
                choices = get_choices(self.iblStudent, doc) 
                dec = userAnnotation['Decision']
                choice, details = self.iblStudent.choose(choices, details=True)
                if(dec == True and choice['Decision'] == 'True' or dec == False and choice['Decision'] == 'False'):
                    self.iblStudent.respond(0.8)
                else:
                    self.iblStudent.respond(0.2)

                if(details[0]['choice']['Decision'] == 'True'):
                    phishing_val = details[0]['blended_value']
                    ham_val = details[1]['blended_value']
                    hamSim = (ham_val) ** 2
                    phishSim = (phishing_val) ** 3
                else:
                    phishing_val = details[1]['blended_value']
                    ham_val = details[0]['blended_value']
                    hamSim = (ham_val) ** 3
                    phishSim = (phishing_val) ** 2
                
                sims = np.array([hamSim, phishSim])
                sims = sims / (np.sum(sims) / 0.8)
                sims += np.random.normal(0,0.05,2)
                sims = np.clip(sims, 0.05, 0.95)

                self.similarities[userAnnotation["EmailId"]].append(sims)
            
    def similarity(self, u, categories=None):
        doc = self.documents[self.documents["Embedding"] == tuple(u)].iloc[0]
        similarity = {}
        sims = self.similarities[doc["EmailId"]]
        if(len(sims) == 0):
            #doc = self.documents[self.documents["EmailId"] == doc["EmailId"]].iloc[0]
            choices = get_choices(self.iblStudent, doc)
            choice, details = self.iblStudent.choose(choices, details=True)

            if(details[0]['choice']['Decision'] == 'True'):
                phishing_val = details[0]['blended_value']
                ham_val = details[1]['blended_value']
                hamSim = (ham_val) ** 2
                phishSim = (phishing_val) ** 3
            else:
                phishing_val = details[1]['blended_value']
                ham_val = details[0]['blended_value']
                hamSim = (ham_val) ** 3
                phishSim = (phishing_val) ** 2
            self.iblStudent.respond(0)

            # balance
            self.similarities[doc["EmailId"].item()].append([hamSim, phishSim])
            sims = self.similarities[doc["EmailId"].item()]
        self.iblStudent = copy.deepcopy(self.baseIblStudent) 
        sims = np.mean(np.array(sims), axis=0)
        for idx, cat in enumerate(self.categories):
            similarity[cat] = sims[idx]

        return similarity

