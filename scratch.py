# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:53:19 2018

@author: Aditya
"""
from gerrychain import Graph, Partition, GeographicPartition
from gerrychain.updaters import Tally,county_splits
import os

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous, no_worse_L1_reciprocal_polsby_popper
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
import gerrychain.scores
from gerrychain import Election
from gerrychain.constraints.validity import deviation_from_ideal

from collections import Counter
import numpy as np


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
                "population": Tally("TOT_POP", alias="population"), 
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
    
    
   

#
#filename = "PA_VTD.shp"
#direc = "PA_VTD"
#fullpath = os.path.join(direc, filename)
#graph = Graph.from_file(fullpath)
#run_simple(graph)
#
#
#p2 = Partition(
#        graph,
#        assignment="2011_PLA_1",
#        updaters={
#            "2016_President": election,
#            "population": Tally("TOT_POP", alias="population"), 
#            "county_split" : county_splits( 'HI', "COUNTYFP10")
#        }
#    )

def equal_split_score(partition, population_name = 'population'):
    '''Take a partition and compute the root mean square deviance from a perfect equal split'''
    deviations = deviation_from_ideal(partition, population_name)
    score = np.linalg.norm(list(deviations.values()))
    return(score)
    
    
def county_split_wrapper(partition, county_split_name = 'county _split', district_name = 'district_name'):
    splits = county_split_score(partition, county_split_name)
    score = compute_countySplitWeight(partition, splits, county_split_name, district_name)
    return(score)
    
def county_split_score(partition, county_split_name = 'county_split'):
    splits = partition[county_split_name]
    to_check = {}
    for s in splits:
        num_split = len(splits[s][2])
        if(num_split > 1):
            prev = to_check.get(num_split, [])
            prev.append(s)
            to_check[num_split] = prev
    return(to_check)

def compute_countySplitWeight(partition, info, county_split_name = 'county_split', district_name = "2011_PLA_1"):
    '''Compute the county score function as described by the paper
    Takes in a parition with a county score_split data and computes the final 
    score'''
    two_counties = info[2]
    two_county_weight_sum = 0
    '''FIxme - should really combine both parts'''
    for c in two_counties:        
        nodes = partition[county_split_name][c][1]
        counties = [partition.graph.nodes[n][district_name] for n in nodes]
        cn = Counter(counties)
        second_frac = cn.most_common(2)[1][1]/ sum(cn.values())
        two_county_weight_sum += np.sqrt(second_frac)
    
    three_plus_weight_sum = 0
    for count_counties in range(3, len(info) +1):
        counties_split_list = info[count_counties]
        for c in counties_split_list:
            nodes = partition[county_split_name][c][1]
            counties = [partition.graph.nodes[n][district_name] for n in nodes]
            cn = Counter(counties)
            temp_frac = 0
            lst = cn.most_common()
            for i in range(2, len(lst)):
                temp_frac += lst[i][1]
            temp_frac = temp_frac / sum(cn.values())
            three_plus_weight_sum += np.sqrt(temp_frac)
    
    # Comput the 2 county split weight vs the 3 
    num_2 = len(two_counties)
    flat_list = [item for sublist in list(info.values()) for item in sublist]
    num_3plus = len(flat_list) - num_2
    final_score = two_county_weight_sum * num_2 + num_3plus * three_plus_weight_sum * 1000
    if(final_score < 0):
        raise Exception("FInal weight should be less than 0")
    return(final_score)
