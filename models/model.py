import manifolds
from layers.lgnn_layer import LorentzGIN
from layers.lorentz_ops import LorentzAct, LorentzLinear, LorentzAgg
from torch_geometric.nn import global_mean_pool
import numpy as np
import torch
from torch.nn import Module, Parameter, Sequential, ReLU, Sigmoid, Softmax


class Classifier(Module):
    def __init__(self, eps, num_layers_mlp, num_classes, c_in, c_out, in_features, dropout, use_att, use_bias):
        super(Classifier, self).__init__()

        self.manifold = getattr(manifolds, 'Lorentzian')()
        self.c_out = c_out
        if num_classes == 2:
            act = Sigmoid()
            self.final_out = 2
        else:
            act = Softmax(dim=1)
            self.final_out = num_classes+1
        self.gin = LorentzGIN(
            manifold=self.manifold,
            eps=eps,
            in_features=in_features,
            c_in=c_in,
            nn=GinMLP(num_layers=num_layers_mlp, c=self.c_out,
                      in_features=in_features, dropout=dropout, use_bias=use_bias),
            use_att=use_att,
            use_bias=use_bias
        )
        self.classifier = LorentzLinear(
            self.manifold, in_features=512-1, out_features=self.final_out, c=self.c_out, dropout=dropout, use_bias=use_bias)
        self.prob = LorentzAct(
            self.manifold, c_in=self.c_out, c_out=self.c_out, act=act)

    def forward(self, input, batch):
        x, edge_index = input

        input = (x, edge_index)
        h = self.gin.forward(input)

        h_tangential = self.manifold.log_map_zero(h, c=self.c_out)
        h_tangential_mean = global_mean_pool(h_tangential, batch)

        h_exp = self.manifold.exp_map_zero(h_tangential_mean, c=self.c_out)
        h_classify = self.classifier(h_exp)

        h_classify_log = self.manifold.log_map_zero(h_classify, c=self.c_out)
        h_classify_prob = self.prob(h_classify)

        # Ignoring origin
        return h_classify_log[:, 1:].view(h_classify_log.size(0),), h_classify_prob[:, 1:].view(h_classify_prob.size(0),)


class GinMLP(Module):
    def __init__(self, num_layers, c, in_features, dropout, use_bias):
        super(GinMLP, self).__init__()
        assert num_layers > 0
        self.c = c
        self.manifold = getattr(manifolds, 'Lorentzian')()

        self.curvatures = [Parameter(torch.tensor([1.]))
                           for _ in range(num_layers)]
        self.curvatures.append(self.c)
        layers = []
        feat = 128
        for i in range(num_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            if i == 0:
                block = Sequential(
                    LorentzLinear(manifold=self.manifold, in_features=in_features-1,
                                  out_features=feat, c=c_in, dropout=dropout, use_bias=use_bias),
                    LorentzAct(manifold=self.manifold, c_in=c_in, c_out=c_out, act=ReLU()))
            else:
                block = Sequential(
                    LorentzLinear(manifold=self.manifold, in_features=feat-1,
                                  out_features=2*feat, c=c_in, dropout=dropout, use_bias=use_bias),
                    LorentzAct(manifold=self.manifold, c_in=c_in,
                               c_out=c_out, act=ReLU())
                )
                feat = feat*2

            layers.append(block)
        self.layers = Sequential(*layers)

    def forward(self, x):
        x = self.layers.forward(x)

        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
