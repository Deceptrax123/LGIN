from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from hyperparameters import EPOCHS, EPSILON, LR, CLIP_VALUE, EPS, NUM_LAYERS_MLP, C_IN, C_OUT, DROPOUT, USE_ATT, USE_BIAS
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
from metrics import classification_binary_metrics, classification_multiclass_metrics
from sklearn.model_selection import train_test_split
from optimizers.radam import RiemannianAdam
from models.model import Classifier
import wandb
import os
from dotenv import load_dotenv
from torch import nn
import torch


def train_epoch():
    epoch_loss = 0
    epoch_acc = 0
    for step, data in enumerate(train_loader):
        if dataset.num_node_features == 0:
            data.x = torch.ones(
                (data.num_nodes, num_in_features), dtype=torch.float32)
        x, adj = data.x, data.edge_index
        input = (x, adj)

        logits, probs = model(input, batch=data.batch)
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
            acc = classification_binary_metrics(probs, data.y.int())
        else:
            acc = classification_multiclass_metrics(
                probs, data.y.int(), dataset.num_classes)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/(step+1), epoch_acc/(step+1)


@torch.no_grad()
def val_epoch():
    epoch_loss = 0
    epoch_acc = 0
    for step, data in enumerate(val_loader):
        if dataset.num_node_features == 0:
            data.x = torch.ones(
                (data.num_nodes, num_in_features), dtype=torch.float32)
        x, adj = data.x, data.edge_index
        input = (x, adj)

        logits, probs = model(input, batch=data.batch)
        loss = loss_function(logits, data.y.float())

        if dataset.num_classes == 2:
            acc = classification_binary_metrics(probs, data.y.int())
        else:
            acc = classification_multiclass_metrics(
                probs, data.y.int(), dataset.num_classes)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        return epoch_loss/(step+1), epoch_acc/(step+1)


@torch.no_grad()
def test():
    test_acc = 0
    for step, data in enumerate(test_loader):
        if dataset.num_node_features == 0:
            data.x = torch.ones(
                (data.num_nodes, num_in_features), dtype=torch.float32)
        x, adj = data.x, data.edge_index
        input = (x, adj)

        logits, probs = model(input, batch=data.batch)
        if dataset.num_classes == 2:
            acc = classification_binary_metrics(probs, data.y.float())
        else:
            acc = classification_multiclass_metrics(
                probs, data.y.int(), dataset.num_classes)

        test_acc += acc.item()
    return test_acc/(step+1)


def training_loop():

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc = train_epoch()
        model.eval()

        with torch.no_grad():
            val_loss, val_acc = val_epoch()

            print("Epoch: ", epoch+1)
            print("Train Loss: ", train_loss)
            print("Train Accuracy: ", train_acc)
            print("Validation Loss: ", val_loss)
            print("Validation Accuracy: ", val_acc)

            wandb.log({
                "Train Loss": train_loss,
                "Train Accuracy": train_acc,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc
            })

            if (epoch+1) % 10 == 0:
                save_path = os.getenv(
                    f"{inp_name}_weights")+f"model_{epoch+1}.pt"
                # Save weights here
                torch.save(model.state_dict(), save_path)

            if (epoch+1) % 50 == 0:
                test_acc = test()
                print(f"Test Accuracy after {epoch+1} epochs is {test_acc}")
                wandb.log({
                    "Test Accuracy": test_acc
                })


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

    dataset.shuffle()
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15

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
    print(dataset.num_node_features)

    model = Classifier(eps=EPS, num_layers_mlp=NUM_LAYERS_MLP, num_classes=dataset.num_classes, c_in=C_IN, c_out=C_OUT, in_features=num_in_features, dropout=DROPOUT, use_att=USE_ATT, use_bias=USE_BIAS
                       )
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
