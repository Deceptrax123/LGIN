from torch_geometric.datasets import TUDataset, MoleculeNet
from torch_geometric.loader import DataLoader
from hyperparameters import EPOCHS, EPSILON, LR, CLIP_VALUE, EPS, NUM_LAYERS_MLP, C_IN, C_OUT, DROPOUT, USE_ATT, USE_BIAS
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
from metrics import classification_binary_metrics, classification_multiclass_metrics
from sklearn.model_selection import train_test_split
from optimizers.radam import RiemannianAdam
from models.model import Classifier, MultilayerGIN
import wandb
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import time
from torch import nn
import torch


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())

    return ave_grads


def train_epoch():
    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0
    for step, data in enumerate(train_loader):
        if dataset.num_node_features == 0:
            data.x = torch.ones(
                (data.num_nodes, num_in_features), dtype=torch.float32)
        x, adj = data.x.float(), data.edge_index
        input = (x, adj)

        # weights = compute_class_weights(data.y)

        optimizer.zero_grad()
        if data.y.size() == (data.y.size(0),):
            data.y = data.y.view(data.y.size(0), 1)
        logits, probs = model(input, batch=data.batch)
        print(logits.size())
        print(data.y.size())

        loss = loss_function(logits, data.y.float())
        # print(logits[:, 1:30])

        loss.backward()

        # Gradient clip
        max_norm = CLIP_VALUE
        all_params = list(model.parameters())
        for param in all_params:
            nn.utils.clip_grad_norm_(param, max_norm)

        # Riemennian Optimization
        optimizer.step()

        if dataset.num_classes == 2:
            acc, auroc = classification_binary_metrics(probs, data.y.int())
        else:
            print(probs.size())
            acc, auroc = classification_multiclass_metrics(
                probs, data.y.int(), dataset.num_classes)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_auc += auroc.item()

    return epoch_loss/(step+1), epoch_acc/(step+1), epoch_auc/(step+1)


@torch.no_grad()
def val_epoch():
    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0
    for step, data in enumerate(val_loader):
        if dataset.num_node_features == 0:
            data.x = torch.ones(
                (data.num_nodes, num_in_features), dtype=torch.float32)
        x, adj = data.x.float(), data.edge_index
        input = (x, adj)
        if data.y.size() == (data.y.size(0),):
            data.y = data.y.view(data.y.size(0), 1)
        logits, probs = model(input, batch=data.batch)

        loss = loss_function(logits, data.y.float())

        if dataset.num_classes == 2:
            acc, auc = classification_binary_metrics(probs, data.y.int())
        else:
            acc, auc = classification_multiclass_metrics(
                probs, data.y.int(), dataset.num_classes)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_auc += auc.item()

    return epoch_loss/(step+1), epoch_acc/(step+1), epoch_auc/(step+1)


@torch.no_grad()
def test():
    test_acc = 0
    test_auc = 0
    for step, data in enumerate(test_loader):
        if dataset.num_node_features == 0:
            data.x = torch.ones(
                (data.num_nodes, num_in_features), dtype=torch.float32)
        x, adj = data.x.float(), data.edge_index
        input = (x, adj)
        data.y = data.y.view(data.y.size(0), 1)
        logits, probs = model(input, batch=data.batch)
        if dataset.num_classes == 2:
            acc, auc = classification_binary_metrics(probs, data.y.int())
        else:
            acc, auc = classification_multiclass_metrics(
                probs, data.y.int(), dataset.num_classes)

        test_acc += acc.item()
        test_auc += auc.item()
    return test_acc/(step+1), test_auc/(step+1)


def training_loop():

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc, train_auc = train_epoch()
        model.eval()

        with torch.no_grad():
            val_loss, val_acc, val_auc = val_epoch()

            print("Epoch: ", epoch+1)
            print("Train Loss: ", train_loss)
            print("Train Accuracy: ", train_acc)
            print("Train AUC: ", train_auc)
            print("Validation Loss: ", val_loss)
            print("Validation Accuracy: ", val_acc)
            print("Validation AUC: ", val_auc)

            grads = plot_grad_flow(model.named_parameters())

            wandb.log({
                "Train Loss": train_loss,
                "Train Accuracy": train_acc,
                "Train AUC": train_auc,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc,
                "Validation AUC": val_auc,
                "Gradients": wandb.Histogram(grads)
            })

            test_acc, test_auc = test()
            print(f"Test Accuracy: {test_acc}")
            print(f"Test AUC: {test_auc}")
            wandb.log({
                "Test Accuracy": test_acc,
                "Test AUC": test_auc
            })

            if (epoch+1) % 10 == 0:
                save_path = os.getenv(
                    f"{inp_name}_weights")+f"/Run_1/model_{epoch+1}.pt"
                # Save weights here
                torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    load_dotenv('.env')
    # torch.autograd.set_detect_anomaly(True)

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

    if inp_name == 'imdb_b':
        dataset = TUDataset(
            root=imdb_b, name='IMDB-BINARY')
    elif inp_name == 'reddit_b':
        dataset = TUDataset(
            root=reddit_b, name='REDDIT-BINARY')
    elif inp_name == 'collab':
        dataset = TUDataset(root=collab, name='COLLAB')
    elif inp_name == 'mutag':
        dataset = TUDataset(root=mutag, name='MUTAG', use_node_attr=True)
    elif inp_name == 'proteins':
        dataset = TUDataset(root=proteins, name='PROTEINS')
    elif inp_name == 'proteins_full':
        dataset = TUDataset(root=proteins_full, name='PROTEINS_full')
    elif inp_name == 'enzymes':
        dataset = TUDataset(root=enzymes, name='ENZYMES')
    elif inp_name == 'clintox':
        dataset = MoleculeNet(root=clintox, name='ClinTox')
    elif inp_name == 'bbbp':
        dataset = MoleculeNet(root=bbbp, name='BBBP',
                              transform=(T.RemoveIsolatedNodes()))
    elif inp_name == 'hiv':
        dataset = MoleculeNet(root=hiv, name='HIV', transform=(
                              T.RemoveIsolatedNodes()))
    elif inp_name == 'sider':
        dataset = MoleculeNet(root=sider, name='SIDER', transform=(
            T.RemoveIsolatedNodes()))

    dataset.shuffle()
    train_ratio = 0.80
    validation_ratio = 0.10
    test_ratio = 0.10

    params = {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 0
    }

    # model instantiation here->
    train_set, test_set = train_test_split(dataset, test_size=1 - train_ratio)
    val_set, test_set = train_test_split(
        test_set, test_size=test_ratio/(test_ratio + validation_ratio))

    train_loader = DataLoader(train_set, **params)
    val_loader = DataLoader(val_set, **params)
    test_loader = DataLoader(test_set, **params)
    if dataset.num_node_features == 0:
        num_in_features = 2
    else:
        num_in_features = dataset.num_node_features

    # model = Classifier(eps=EPS, num_layers_mlp=NUM_LAYERS_MLP, num_classes=dataset.num_classes, c_in=C_IN, c_out=C_OUT, in_features=num_in_features, dropout=DROPOUT, use_att=USE_ATT, use_bias=USE_BIAS
    #                    )
    model = MultilayerGIN(eps=EPS, num_layers_mlp=NUM_LAYERS_MLP, num_classes=dataset.num_classes, c_in=C_IN,
                          c_out=C_OUT, in_features=num_in_features, dropout=DROPOUT, use_att=USE_ATT, use_bias=USE_BIAS)
    optimizer = RiemannianAdam(
        params=model.parameters(), lr=LR, weight_decay=EPSILON)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=50)
    if dataset.num_classes == 2:
        loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    wandb.init(
        project="Lorentzian Graph Isomorphism Network"
    )

    training_loop()
