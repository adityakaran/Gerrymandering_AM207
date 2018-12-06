# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:53:19 2018

@author: Aditya
"""
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally
import os

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous, no_worse_L1_reciprocal_polsby_popper
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
import gerrychain.scores
from gerrychain import Election



def generate_graph(path):
    return Graph.from_file(path)


def run_simple(graph):
    election = Election(
        "2016 President",
        {"Democratic": "T16PRESD", "Republican": "T16PRESR"},
        alias="2016_President"
    )
    
    
    initial_partition = Partition(
            graph,
            assignment="2011_PLA_1",
            updaters={
                "2016_President": election,
                "population": Tally("TOT_POP", alias="population")
            }
        )
        
    
    efficiency_gaps = []
    wins = []
    
    chain = MarkovChain(
        proposal=propose_random_flip,
        is_valid=single_flip_contiguous,
        accept=always_accept, #THe acceptance criteria is what needs to be defined ourselves - to match the paper
        initial_state=initial_partition,
        total_steps=100000
    )
    
    for partition in chain:
        efficiency_gaps.append(gerrychain.scores.efficiency_gap(partition["2016_President"]))
        wins.append(partition["2016_President"].wins("Democratic"))
    
    
    
filename = "PA_VTD.shp"
direc = "PA_VTD"
fullpath = os.path.join(direc, filename)
graph = Graph.from_file(fullpath)
run_simple(graph)

