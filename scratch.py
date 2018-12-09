# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:53:19 2018

@author: Aditya
"""
from gerrychain import Graph, Partition, GeographicPartition
from gerrychain.updaters import Tally, county_splits, boundary_nodes, cut_edges, cut_edges_by_part, exterior_boundaries, interior_boundaries, perimeter
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
import random

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
                "perimeter": perimeter,
                "exterior_boundaries": exterior_boundaries,
                "interior_boundaries": interior_boundaries,
                "boundary_nodes": boundary_nodes,
                "cut_edges": cut_edges,
                "area": Tally("area", alias="area"),
                "cut_edges_by_part": cut_edges_by_part, 
                "county_split" : county_splits( 'county_split', "COUNTYFP10"),
            
            }
        )
        
    
    efficiency_gaps = []
    wins = []
    
    chain = MarkovChain(
        proposal=propose_random_flip,
        is_valid=single_flip_contiguous,
        accept=always_accept, #THe acceptance criteria is what needs to be defined ourselves - to match the paper
        initial_state=initial_partition,
        total_steps=10
    )
    
    for partition in chain:
        efficiency_gaps.append(gerrychain.scores.efficiency_gap(partition["2016_President"]))
        wins.append(partition["2016_President"].wins("Democratic"))
    
    
    
   


#p2 = Partition(
#        graph,
#        assignment="2011_PLA_1",
#        updaters={
#            "2016_President": election,
#            "population": Tally("TOT_POP", alias="population"), 
#            "black_population": Tally("BLACK_POP", alias = "black_population"),
#            "county_split" : county_splits( 'HI', "COUNTYFP10"),
#                            "perimeter": perimeter,
#                "exterior_boundaries": exterior_boundaries,
#                "interior_boundaries": interior_boundaries,
#                "boundary_nodes": boundary_nodes,
#                "cut_edges": cut_edges,
#                "area": Tally("area", alias="area"),
#                "cut_edges_by_part": cut_edges_by_part
#
#        }
#    )


def metro_scoring_prob(partition, beta, wp, wi, wc, wm):
    if(partition.parent == None):
        return True
    conflictedp = len(partition['cut_edges'])
    conflicted = conflictedp
    conflicted = len(partition.parent['cut_edges'])
    ratio = conflicted/conflictedp
    curr_score = score_partition(partition, wp, wi, wc, wm)
    prev_score = score_partition(partition.parent, wp, wi, wc, wm)
    exp = np.exp(-beta * (curr_score - prev_score))
    prob = min(1, ratio * exp)
    return(prob > random.random())
    
    
def score_partition(partition, wp, wi, wc, wm):
    return(equal_split_score(partition) * wp + compactness_split_score(partition) + 
           county_split_wrapper(partition) * wc + vra_district_requirement(partition) * wm)

def vra_district_requirement(partition, num_districts = 2, thresholds = [0.445, 0.36]):
    if(len(thresholds) != num_districts):
        raise Exception("Number of thresholds needs to equal the number of districts you want")
    black_pop_dict = partition['black_population']
    total_dict = partition['population']
    fractions = Counter({k : black_pop_dict[k] / total_dict[k] for k in total_dict})
    print(fractions)
    top_n = fractions.most_common(num_districts)
    thresholds.sort()
    print(top_n)
    score = 0
    for i in range(0, num_districts):
        #Get the max thresholds
        temp_score = max(0, thresholds[i] - top_n[i][1])
        score += np.sqrt(temp_score)
    return(score)
    
    
    
    
def isoparametric(area, perimeter):
    try:
        return perimeter**2/area
    except ZeroDivisionError:
        return np.nan

def polsby_popper(area, perimeter):
    try:
        return 4*np.pi*area / perimeter**2
    except ZeroDivisionError:
        return np.nan

def compact_dispersion(area, perimeter):
    raise NotImplementedError

def compactness_split_score(partition):
    isoparametric_parts = []
    #polsby_popper_parts = []
    #dispersion_parts = []
    for part in partition.parts:
        area, perimeter = partition['area'][part], partition['perimeter'][part]
        isoparametric_parts.append(isoparametric(area, perimeter))
        #polsby_popper_parts.append(polsby_popper(area, perimeter))
        #dispersion_parts.append()
    return np.sum(np.array(isoparametric_parts))
    #return np.array(polsby_popper_parts)
    #return np.array(dispersion_parts)

def equal_split_score(partition, population_name = 'population'):
    '''Take a partition and compute the root mean square deviance from a perfect equal split'''
    deviations = deviation_from_ideal(partition, population_name)
    score = np.linalg.norm(list(deviations.values()))
    return(score)
    
    
def county_split_wrapper(partition, county_split_name = 'county_split', district_name = '2011_PLA_1'):
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

if __name__ == "__main__":
    graph = generate_graph(os.path.join("PA_VTD", 'PA_VTD.shp'))
    run_simple(graph)