import numpy as np

from question_filter import *
from multimedia import *
from question_decoder_3 import *
from recommender_system_emb_0 import *

def run_driver(question, graph, data):
    filter_result = run_filter(question)

    if filter_result == "Multimedia":
        response = run_mult(question, graph, data)
    
    elif filter_result == "Recommender":
        response = run_recm(question, graph)

    else:
        response = run_qdec(question, graph)
    
    return response
