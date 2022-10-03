# HTC
Code for the paper "[Towards Higher-order Topological Consistency for Unsupervised Network Alignment](https://arxiv.org/pdf/2208.12463.pdf)".

# Environment
- pytorch >= 1.11.0
- networkx >= 2.6.0

# Running
- Douban Online & Offline

'''
python network_alignment.py --source_dataset graph_data/douban/online/graphsage --target_dataset graph_data/douban/offline/graphsage --groundtruth graph_data/douban/dictionaries/groundtruth HTC --k 20 --p 0.5 --ulr 0.01 --alpha 1.1
'''
- Allmovie & Imdb

'''
python network_alignment.py --source_dataset graph_data/allmv_tmdb/allmv/graphsage --target_dataset graph_data/allmv_tmdb/tmdb/graphsage --groundtruth graph_data/allmv_tmdb/dictionaries/groundtruth HTC --gm
'''
