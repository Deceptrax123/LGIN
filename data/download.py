from torch_geometric.datasets import TUDataset
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

    # zero feature
    transform = T.OneHotDegree(max_degree=4)

    if inp_name == 'imdb_b':
        dataset = TUDataset(
            root=imdb_b, name='IMDB-BINARY', use_node_attr=True, transform=transform)
    elif inp_name == 'reddit_b':
        dataset = TUDataset(
            root=reddit_b, name='REDDIT-BINARY', use_node_attr=True, transform=transform)
    elif inp_name == 'collab':
        dataset = TUDataset(root=collab, name='COLLAB',
                            use_node_attr=True, transform=transform)
    elif inp_name == 'mutag':
        dataset = TUDataset(root=mutag, name='MUTAG', use_node_attr=True)
    elif inp_name == 'proteins':
        dataset = TUDataset(root=proteins, name='PROTEINS', use_node_attr=True)
    elif inp_name == 'proteins_full':
        dataset = TUDataset(root=proteins_full, name='PROTEINS_full')


main()
