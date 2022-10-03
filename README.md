# HTC
Code for the paper "[Towards Higher-order Topological Consistency for Unsupervised Network Alignment](https://arxiv.org/pdf/2208.12463.pdf)".

# Environment
- pytorch >= 1.11.0
- networkx >= 2.6.0

# Running
For orbit counting on new datasets, please refer to [ORCA](https://github.com/thocevar/orca).
Before running, make sure that the file "graph_data" (see **Releases**) is downloaded and uncompressed.
Then run the following codes:

- Douban Online & Offline

```php
python network_alignment.py --source_dataset graph_data/douban/online/graphsage --target_dataset graph_data/douban/offline/graphsage --groundtruth graph_data/douban/dictionaries/groundtruth HTC --k 20 --p 0.5 --ulr 0.01 --alpha 1.1
```
- Allmovie & Imdb

```php
python network_alignment.py --source_dataset graph_data/allmv_tmdb/allmv/graphsage --target_dataset graph_data/allmv_tmdb/tmdb/graphsage --groundtruth graph_data/allmv_tmdb/dictionaries/groundtruth HTC --gm
```

# Citation

Please kindly cite our work as follows:

*Qingqiang Sun, Xuemin Lin, Ying Zhang, Wenjie Zhang, Chaoqi Chen. Towards Higher-order Topological Consistency for Unsupervised Network Alignment. arXiv preprint arXiv:2208.12463, 2022.*
