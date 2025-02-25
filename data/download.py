from torch_geometric.datasets import TUDataset, MoleculeNet
import torch_geometric.transforms as T
import os
from dotenv import load_dotenv


def main():
    load_dotenv('.env')

    inp_name = input('Enter the dataset to be downloaded: ')

    imdb_b = os.getenv('imdb_b')
    reddit_b = os.getenv('reddit_b')
    collab = os.getenv('collab')
    mutag = os.getenv('mutag')
    proteins = os.getenv('proteins')
    proteins_full = os.getenv('proteins_full')
    enzymes = os.getenv('enzymes')
    clintox = os.getenv('clintox')
    bbbp = os.getenv('bbbp')
    hiv = os.getenv('hiv')
    sider = os.getenv('sider')
    tox21 = os.getenv('tox21')
    bace = os.getenv('bace')

    if inp_name == 'imdb_b':
        dataset = TUDataset(
            root=imdb_b, name='IMDB-BINARY', transform=(T.OneHotDegree(max_degree=10)))
    elif inp_name == 'reddit_b':
        dataset = TUDataset(
            root=reddit_b, name='REDDIT-BINARY')
    elif inp_name == 'collab':
        dataset = TUDataset(root=collab, name='COLLAB')
    elif inp_name == 'mutag':
        dataset = TUDataset(root=mutag, name='MUTAG', use_node_attr=True)
    elif inp_name == 'proteins':
        dataset = TUDataset(root=proteins, name='PROTEINS', use_node_attr=True)
    elif inp_name == 'proteins_full':
        dataset = TUDataset(root=proteins_full, name='PROTEINS_full')
    elif inp_name == 'enzymes':
        dataset = TUDataset(root=enzymes, name='ENZYMES')
    elif inp_name == 'clintox':
        dataset = MoleculeNet(root=clintox, name='ClinTox')
    elif inp_name == 'bbbp':
        dataset = MoleculeNet(root=bbbp, name='BBBP')
    elif inp_name == 'hiv':
        dataset = MoleculeNet(root=hiv, name='HIV')
    elif inp_name == 'sider':
        dataset = MoleculeNet(root=sider, name='SIDER')
    elif inp_name == 'tox21':
        dataset = MoleculeNet(root=tox21, name='Tox21')
    elif inp_name == 'bace':
        dataset = MoleculeNet(root=bace, name='BACE')


main()
