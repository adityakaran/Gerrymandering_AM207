# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:53:19 2018

@author: Aditya
"""
from gerrychain import Graph, Partition, GeographicPartition
from gerrychain.updaters import Tally, county_splits, boundary_nodes, cut_edges, cut_edges_by_part, exterior_boundaries, interior_boundaries, perimeter
import os

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous, no_worse_L1_reciprocal_polsby_popper, districts_within_tolerance
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
import gerrychain.scores
from gerrychain import Election
from gerrychain.constraints.validity import deviation_from_ideal
from gerrychain.constraints import Validator
from collections import Counter
import numpy as np
import random
from descartes import PolygonPatch
import matplotlib.pyplot as plt

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
                "black_population": Tally("BLACK_POP", alias = "black_population"),
            
            }
        )
        
    
    efficiency_gaps = []
    wins = []
    beta = 0
    wp = 1
    wi = 1
    wc = 0
    wm = 1
    def accept(partition):
       return(metro_scoring_prob(partition, beta, wp, wi, wc, wm))
   
    is_valid = Validator([single_flip_contiguous, districts_within_tolerance ])
    chain = MarkovChain(
        proposal=propose_random_flip,
        is_valid=is_valid,
        #accept = always_accept,
        accept=accept, #THe acceptance criteria is what needs to be defined ourselves - to match the paper
        initial_state=initial_partition,
        total_steps=30
    )
    
    for partition in chain:
        if(hasattr(partition, 'accepted') and partition.accepted):
            efficiency_gaps.append(gerrychain.scores.efficiency_gap(partition["2016_President"]))
            wins.append(partition["2016_President"].wins("Democratic"))
    return(efficiency_gaps, wins, partition)
    
    
    
   


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
    top_n = fractions.most_common(num_districts)
    thresholds.sort()
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
    
    
def county_split_wrapper(partition, county_split_name = 'county_split', district_name = 'county_split'):
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

def compute_countySplitWeight(partition, info, county_split_name = 'county_split', district_name = "county_split"):
    '''Compute the county score function as described by the paper
    Takes in a parition with a county score_split data and computes the final 
    score'''
    two_counties = info[2]
    two_county_weight_sum = 0
    '''FIxme - should really combine both parts'''
    for c in two_counties:        
        nodes = partition[county_split_name][c][1]
        counties = [partition.assignment[n] for n in nodes]
        cn = Counter(counties)
        second_frac = cn.most_common(2)[1][1]/ sum(cn.values())
        two_county_weight_sum += np.sqrt(second_frac)
    
    three_plus_weight_sum = 0
    to_consider = list(info.keys())
    to_consider.remove(2)
    for count_counties in to_consider:
        counties_split_list = info[count_counties]
        for c in counties_split_list:
            nodes = partition[county_split_name][c][1]
            counties = [partition.assignment[n] for n in nodes]
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
    # Says to use large number for three_plus_weight...
    final_score = two_county_weight_sum * num_2 + num_3plus * three_plus_weight_sum * 100
    if(final_score < 0):
        raise Exception("FInal weight should be less than 0")
    return(final_score)

def plot_state(partition):
    plt.figure()
    ax = plt.axes()
    b = (float('inf'), float('inf'), float('-inf'), float('-inf'))

    colors = get_spaced_colors(len(partition.parts))
    nodes = partition.graph.nodes
    for n in nodes:
        data = partition.graph.nodes[n]
        poly = data['geometry']
        patch = PolygonPatch(poly, facecolor = colors[partition.assignment[n] - 1])
        ax.add_patch(patch)
        b= get_bounds(poly,b )

        
    plt.xlim(b[0], b[2]) #repsent longtitude
    plt.ylim(b[1], b[3]) # represent  lattitude

    plt.show()
        
    
def get_spaced_colors(n):
    '''from quora https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python'''
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    
    return [(int(i[:2], 16)/256., int(i[2:4], 16)/256., int(i[4:], 16)/256.) for i in colors]

def plot_graph(graph):
    plt.figure()
    ax = plt.axes()
    #colors = get_spaced_colors(len(partition.parts))
    b = (float('inf'), float('inf'), float('-inf'), float('-inf'))
    nodes = graph.nodes
    for n in nodes:
        data = graph.nodes[n]
        poly = data['geometry']
        patch = PolygonPatch(poly)
        ax.add_patch(patch)
        b= get_bounds(poly,b )
    plt.xlim(b[0], b[2]) #repsent longtitude
    plt.ylim(b[1], b[3]) # represent  lattitude

    plt.show()

def get_bounds(d, orig):
    (minx, miny, maxx, maxy) = orig
    bounds = d.bounds
    minx = min(bounds[0], minx)
    miny = min(bounds[1], miny)
    maxx = max(bounds[2], maxx)
    maxy = max(bounds[3], maxy)
    return(minx, miny, maxx, maxy)

def run_simple2(graph):
    election = Election(
        "2014 Senate",
        {"Democratic": "sen_blue", "Republican": "sen_red"},
        alias="2014_Senate"
    )
    
    
    initial_partition = Partition(
            graph,
            assignment="con_distri",
            updaters={
                "2014_Senate": election,
                "population": Tally("population", alias="population"), 
                "exterior_boundaries": exterior_boundaries,
                "interior_boundaries": interior_boundaries,
                "perimeter": perimeter,
                "boundary_nodes": boundary_nodes,
               "cut_edges": cut_edges,
                "area": Tally("area", alias="area"),
               "cut_edges_by_part": cut_edges_by_part, 
                "county_split" : county_splits( 'county_split', "COUNTY_ID"),
                "black_population": Tally("black_pop", alias = "black_population"),
            
            }
        )
    districts_within_tolerance_2 = lambda part : districts_within_tolerance(part, 'population', 0.3)
    is_valid = Validator([single_flip_contiguous, districts_within_tolerance_2 ])

    chain = MarkovChain(
        proposal=propose_random_flip,
        is_valid=is_valid,
        accept = always_accept,
   #     accept=accept, #THe acceptance criteria is what needs to be defined ourselves - to match the paper
        initial_state=initial_partition,
        total_steps=30,
    )
    efficiency_gaps = []
    wins = []

    for partition in chain:
        if(hasattr(partition, 'accepted') and partition.accepted):
            efficiency_gaps.append(gerrychain.scores.efficiency_gap(partition["2014_Senate"]))
            wins.append(partition["2014_Senate"].wins("Democratic"))

    return(efficiency_gaps, wins, partition)

graph = generate_graph(os.path.join("test_file", "test_file.shp"))

print(graph.nodes[2293]['con_distri'] )
print(graph.nodes[2349]['con_distri'] )

#Think the data has some error because of how close the stuff is, makes it 
# hard to compress nicely...or it is just that bad
graph.nodes[2293]['con_distri'] = 2
graph.nodes[2349]['con_distri']  = 4

#equal_split_score(init)
#graph.nodes[2293]['con_distri'] = 4
#graph.nodes[2349]['con_distri'] = 4
(eff, wins, part) = run_simple2(graph)


#
#if __name__ == "__main__":
#    graph = generate_graph(os.path.join("PA_VTD", 'PA_VTD.shp'))
#    run_simple(graph)