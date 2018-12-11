# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:05:09 2018

@author: Aditya
"""
import pandas as pd
import geopandas as gpd


def split_p_to_congressional(reduced):
    mini = reduced[reduced['con_districts'] > 13]
    mini['cdist1'] = mini['con_districts'] // 100
    mini['cdist2'] = mini['con_districts'] % 100
    
    #f = pd.concat(
    #        [pd.Series([row['con_districts'] // 100 , 4]) for _, row in mini.iterrows()]).reset_index()
    #, row['con_districts'] // 100, row['con_districts'] % 100
    
    c4 = reduced[reduced['con_districts'] < 14]
    c4 = c4.groupby(by = ["COUNTY_ID", "con_districts"]).agg({'population' : 'sum'})
    
    c4 = c4.groupby(level=0).apply(lambda x:
                                                     100 * x / float(x.sum()))
    c4 = c4.reset_index()
    
    def frac1(row):
        fracs = c4[c4['COUNTY_ID'] == row["COUNTY_ID"]]
        frac_1 = float(fracs[fracs['con_districts'] == row['cdist1']]['population'])
        frac_2 = float(fracs[fracs['con_districts'] == row['cdist2']]['population'])
        return( frac_1 / (frac_1 + frac_2))
    
    def frac2(row):
        fracs = c4[c4['COUNTY_ID'] == row["COUNTY_ID"]]
        frac_1 = float(fracs[fracs['con_districts'] == row['cdist1']]['population'])
        frac_2 = float(fracs[fracs['con_districts'] == row['cdist2']]['population'])
        return( frac_2 / (frac_1 + frac_2))
    
        
    mini['frac1'] = mini.apply(frac1, axis = 1)
    mini['frac2'] = mini.apply(frac2, axis = 1)
    
    pivoted = pd.melt(mini, id_vars = ['PREC_ID', 'COUNTY_ID', 'COUNTY_NAM', 'sen_red', 'sen_blue', 
                             'population', 'white_pop', 'black_pop', 'asian_pop', 'hispanic_pop', 'other_pop', 'con_districts', 'frac1', 'frac2'])
    
    population_cols = [ 'population', 'white_pop' , 'black_pop', 'asian_pop', 'hispanic_pop', 'other_pop', 'sen_red', 'sen_blue']
    
    def choose_frac(row):
        if(row['variable'] == 'cdist1'):
            return(row['frac1'])
        else:
            return(row['frac2'])
    
    pivoted['frac'] = pivoted.apply(choose_frac, axis = 1)
    pivoted[population_cols] = pivoted[population_cols].multiply(pivoted['frac'], axis = 'index')
    pivoted[population_cols] = pivoted[population_cols].round()
    pivoted = pivoted.drop(columns = ['frac1', 'frac2', 'variable', 'frac', 'con_districts'])
    pivoted = pivoted.rename(index = str, columns = {"value": "con_districts"})
    return(pivoted)



states = gpd.read_file(fname)
fname = os.path.join("North-Carolina-2014.geojson.gz")
p2014_geo = gpd.read_file(os.path.join("NC_2014", 'PRECINCTS.shp'))
graph = generate_graph(os.path.join("test_file", "test_file.shp"))

reduced = states[['PREC_ID', 'COUNTY_ID', 'COUNTY_NAM', 'sen_red', 'sen_blue', 'con_districts', 
               'population', 'white_pop', 'black_pop', 'asian_pop', 'hispanic_pop', 'other_pop']]                                                     
print(set(reduced.con_districts))
combined = reduced.groupby(by = ['PREC_ID', "COUNTY_ID", 'COUNTY_NAM', 'con_districts']).sum()
combined = combined.reset_index()
single_districts = combined[combined['con_districts'] < 14]

splitted = split_p_to_congressional(combined)
clean_splitted = pd.concat([single_districts, splitted])

with_geo = p2014_geo.merge(clean_splitted, on = ["PREC_ID", "COUNTY_ID"])
