import argparse
import numpy as np
import os
import json
import networkx as nx
from networkx.readwrite import json_graph

def parse_args():
    parser = argparse.ArgumentParser(description="Generate grountruth self dictionary for econ/bn, that id is invariant")
    parser.add_argument('--input_dir', default="./graph_data/bn/graphsage", help="graph file")
    parser.add_argument('--out_dir', default="./graph_data/bn/dictionaries", help="Output file")
    return parser.parse_args()



def generate_and_save(input_dir, out_dir):
    """
    nodes: list of nodes' ids
    out_file: directory to save output files.
    """
    input_dir += "/"
    id2idx_file = os.path.join(input_dir, 'id2idx.json')
    id2idx = json.load(open(id2idx_file))
    idx2id = {v: k for k, v in id2idx.items()}
    G_data = json.load(open(input_dir + "G.json"))

    G_data['links'] = [
        {'source': idx2id[G_data['links'][i]['source']], 'target': idx2id[G_data['links'][i]['target']]} for i
        in range(len(G_data['links']))]
    G = json_graph.node_link_graph(G_data)
    print(nx.info(G))

    n = len(G.nodes())

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    outfile = open(os.path.join(out_dir, 'groundtruth'), 'w+')
    for node in G.nodes():
        print(node)
        outfile.write("{0} {1}".format(node, node))
        outfile.write('\n')
    outfile.close()


if __name__ == '__main__':
    args = parse_args()

    generate_and_save(args.input_dir, args.out_dir)