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
                