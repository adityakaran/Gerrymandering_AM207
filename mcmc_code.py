# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 03:51:14 2018

@author: Aditya
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:53:19 2018

@author: Aditya
"""
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally, county_splits, boundary_nodes, cut_edges, cut_edges_by_part, exterior_boundaries, interior_boundaries, perimeter
import os

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous, districts_within_tolerance
from gerrychain.proposals import propose_random_flip
import gerrychain.scores
from gerrychain import Election
from gerrychain.constraints.validity import deviation_from_ideal
from gerrychain.constraints import Validator
from collections import Counter
import numpy as np
import random
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import time

class MarkovChainAnneal(MarkovChain):
    '''We have a different step function since we want annealing. This is taken from 
    GerryChain library and modified as apprpriate'''
    def __next__(self):
        if self.counter == 0:
            self.counter += 1
            return self.state

        while self.counter < self.total_steps:
            proposal = self.proposal(self.state)

            if not proposal:
                if self.accept(self.state, self.counter):
                    self.counter += 1
                    return self.state
                else:
                    continue

            proposed_next_state = self.state.merge(proposal)
            # Erase the parent of the parent, to avoid memory leak
            self.state.parent = None

            if self.is_valid(proposed_next_state):
                proposed_next_state.accepted = self.accept(proposed_next_state, self.counter)
                if proposed_next_state.accepted:
                    self.state = proposed_next_state
                self.counter += 1
                # Yield the proposed state, even if not accepted
                return proposed_next_state
        raise StopIteration


def generate_graph(path):
    return Graph.from_file(path)


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
    thresholds.sort(reverse = True)
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
    for part in partition.parts:
        area, perimeter = partition['area'][part], partition['perimeter'][part]
        isoparametric_parts.append(isoparametric(area, perimeter))
    return np.sum(np.array(isoparametric_parts))

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

def plot_state(partition, show_node_boundaries = False):
    plt.figure()
    ax = plt.axes()
    b = (float('inf'), float('inf'), float('-inf'), float('-inf'))
    lw = 0
    if(show_node_boundaries):
        lw = None
    colors = get_spaced_colors(len(partition.parts))
    nodes = partition.graph.nodes
    for n in nodes:
        data = partition.graph.nodes[n]
        poly = data['geometry']
        patch = PolygonPatch(poly, linewidth = lw, facecolor = colors[int(partition.assignment[n]) - 1])
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

def run_simple(graph, num_samples = 80000, get_parts = 5):
    election = Election(
        "2014 Senate",
        {"Democratic": "sen_blue", "Republican": "sen_red"},
        alias="2014_Senate"
    )
    
    
    
    initial_partition = Partition(
            graph,
            assignment="2014_CD",
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
    
    def vra_validator(part):
        if(vra_district_requirement(part, 2, [0.4, 0.335]) > 0):
            return False
        return True

    districts_within_tolerance_2 = lambda part : districts_within_tolerance(part, 'population', 0.12)
    is_valid = Validator([single_flip_contiguous, districts_within_tolerance_2, vra_validator ])
    wp = 3000
    wi = 2.5
    wc = 0.4
    wm = 800
    def accept(partition, counter):
        if(counter < 40000):
            beta = 0
        elif(counter < 60000):
            beta = (counter - 40000) /(60000-40000)
        else:
            beta = 1
        return(metro_scoring_prob(partition, beta, wp, wi, wc, wm))

    chain = MarkovChainAnneal(
        proposal=propose_random_flip,
        is_valid=is_valid,
        accept=accept,
        initial_state=initial_partition,
        total_steps= num_samples * 100,
    )
    # going to show 5 partitions from this sample
    part_con = max(1, num_samples / get_parts)
    
    efficiency_gaps = []
    wins = []
    voting_percents = []
    sample_parts = []
    tbefore = time.time()
    for partition in chain:
        if(hasattr(partition, 'accepted') and partition.accepted):
            efficiency_gaps.append(gerrychain.scores.efficiency_gap(partition["2014_Senate"]))
            wins.append(partition["2014_Senate"].wins("Democratic"))
            voting_percents.append(partition["2014_Senate"].percents("Democratic"))
            num_s = len(wins)
            if(num_s % part_con) == 0:
                sample_parts.append(partition)
            if(num_s> num_samples):
                break
            if(num_s % 1000 == 0):
                tnext = time.time()
                print("It took %i time to get 1000 samples" % (tnext - tbefore))
                tbefore = time.time()
    return(efficiency_gaps, wins, voting_percents, sample_parts)
    
def rep_index(win_percents):
    '''Takes in the demoncratic win_percents. Does a linear fit'''
    a = np.array(win_percents)
    a.sort()
    least_rep = 1 - a[np.argmax(a > 0.5)]
    least_dem = a[np.argmin(a < 0.5) + 1]
    num_win = len(a) - np.argmax(a > 0.5)
    delta = (50 - (100 - least_dem ))/(least_rep - (100 - least_dem))
    score = num_win + delta
    return(score)

def gerrymandering_index(average_sample, each_sample):
    a = np.array(average_sample)
    b = np.array(each_sample)
    a.sort()
    b.sort()
    return(np.linalg.norm(a - b))

def avg_sample(win_percents_arr):
    order = {}
    for i in range(1, 1 + len(win_percents_arr[0])):
        order[i] = 0
    for j in range(len(win_percents_arr)):
        elem = list(win_percents_arr[j])
        elem.sort()
        for s in range(len(elem)):
            order[s + 1] += elem[s]
    for i in range(1, 1 + len(win_percents_arr[0])):
        order[i] *= 1/len(win_percents_arr)
    fin = list(order.values())
    fin.sort(reverse = True)
    return(fin)


#graph = generate_graph(os.path.join("test_file", "test_file.shp"))
#print(graph.nodes[2293]['2016_CD'] )
#print(graph.nodes[2349]['2016_CD'] )
#
#graph.nodes[2293]['2014_CD'] = 2
#graph.nodes[2349]['2014_CD']  = 4
#
#(eff, wins, part) = run_simple2(graph)
#

