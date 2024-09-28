#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com

import os
import sys
import json
import pickle
import numpy as np
import networkx as nx
import csv
from multiprocessing import Pool
from lib.ci import collective_influence


def vg_ci(_input):
    file_, PI, T = _input
    graph_file = os.path.join('../VG', PI, file_)
    with open(graph_file, 'rb') as fp:
        vgs = pickle.load(fp)
    cis = {}
    for d, adj in vgs.items():
        labels = np.array([str(i) for i in range(7)])
        G = nx.Graph()
        for i in range(min(T, adj.shape[0])):  # Ensure 'i' does not exceed 'adj' dimensions
            vg_adjs = labels[np.where(adj[i] == 1)]
            edges = list(zip(*[[labels[i]]*len(vg_adjs), vg_adjs]))
            G.add_edges_from(edges)
        cis[d] = collective_influence(G)
    return cis


if __name__ == '__main__':
    vol_price = ['Open','High','Low','Close','Volume']
    T = 7
    for PI in vol_price:
        vg_dir = os.path.join('../VG/', PI)
        ci_dir = os.path.join('../CI', PI)
        if not os.path.exists(ci_dir):
            os.makedirs(ci_dir)
        pool = Pool()
        results = pool.map(vg_ci, [(f, PI, T) for f in os.listdir(vg_dir)])
        pool.close()
        pool.join()

        # Collect results into a single dictionary
        combined_results = {}
        for result in results:
            combined_results.update(result)
        
        # Write results to a CSV file
        csv_file = os.path.join(ci_dir, 'collective_influence.csv')
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = ['day'] + [str(i) for i in range(7)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for day, ci_values in combined_results.items():
                row = {'day': day}
                row.update(ci_values)
                writer.writerow(row)
