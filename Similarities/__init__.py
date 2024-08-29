from .human import Human 
from .cosine import Cosine
from .semantic import Semantic 
from .ensemble import Ensemble
from .ibis import IBIS
from .custom import Custom
"""
To add new custom metrics either add a new object called MyNewCustomSimilarity (or whatever you want) in the custom.py file and edit the above line to include this metric:  
from .custom import Custom, MyNewCustomSimilarity
or add a new custom file based on the custom.py file and add the line:
from .mynewcustomsimilarity import MyNewCustomSimilarity
"""