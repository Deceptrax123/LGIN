# Lorentzian Graph Isomorphic Network
Official code of our paper "Lorentzian Graph Isomorphic Network". The preprint can be found <a href="https://arxiv.org/abs/2504.00142">here</a>.

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
