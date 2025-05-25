# Lorentzian Graph Isomorphic Network
Official code of our paper Lorentzian Graph Isomorphic Network accepted at ACM SIGMOD 2025 GRADES NDA Workshop. The preprint can be found <a href="https://arxiv.org/abs/2504.00142">here</a>.

## Description
While graph neural networks (GNNs) operating in hyperbolic spaces have shown promise
for modeling hierarchical and complex relational data, a critical limitation often overlooked
is their potentially limited discriminative power compared to their Euclidean counterparts
or fundamental graph isomorphism tests like the Weisfeiler-Lehman (WL) hierarchy. Ex-
isting hyperbolic aggregation schemes, while curvature-aware, may not sufficiently capture
the intricate structural differences required to robustly distinguish non-isomorphic graphs
owing to non-injective aggregation functions. To address this expressiveness gap in hyper-
bolic graph learning, we introduce the Lorentzian Graph Isomorphic Network (LGIN), a
novel GNN designed to achieve enhanced discriminative capabilities within the Lorentzian
model of hyperbolic space. LGIN proposes a new update rule that effectively combines
local neighborhood information with a richer representation of graph structure designed to
preserve the Lorentzian metric tensor. A key theoretical contribution is demonstrating that
LGIN possesses discriminative power at least approximately equivalent to the 1-dimensional
Weisfeiler-Lehman test. This represents a significant step towards building more expressive
GNNs in non-Euclidean geometries, overcoming a common bottleneck in current hyperbolic
methods. We conduct extensive evaluations across nine diverse benchmark datasets, in-
cluding molecular and protein structures. LGIN consistently outperforms or matches state-
of-the-art hyperbolic and Euclidean GNNs, showcasing its practical efficacy and validating
its superior ability to capture complex graph structures and distinguish between different
graphs. To the best of our knowledge, LGIN is the first work to successfully adapt the
core principles behind powerful, highly-discriminative GNN architectures to a Riemannian
manifold

## Running the Scripts
- Create a Python(preferrably >3.10) virtual environment and activate it. Run the following command
```sh
pip install -r requirements.txt
```
- Create a ```.env``` file and set the values of all environment variables. 
- Add dataset root directory paths in the ```.env``` file.
- Model configurations such as epsilon value, curvature, dropout, batch size etc can be edited in ```model_config.py```.
- For training on graph classification, run ```train.py``` and run ```train_regression.py``` for regression.

## Datasets
```sh
Mutag
Proteins
PTC
NCI1
Reddit
IMDB
DD
Enzymes
HIV
BBBP
BACE
Zinc
AQSOL
```

## Main Dependencies
```sh
torch
python-dotenv
torch-geometric
torchmetrics
scikit-learn
numpy
scipy
python-dotenv
matplotlib
wandb
os
```

## Bugs/Queries
If there are any bugs, you may raise an issue. If there are any technical questions regarding the project, you may email me at smudge0110@icloud.com. Contributions are welcome : )
