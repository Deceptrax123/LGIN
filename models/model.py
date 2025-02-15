import manifolds
from layers.lgnn_layer import LorentzGIN
from layers.lorentz_ops import LorentzAct, LorentzLinear, LorentzAgg
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
        else:
            act = Softmax()
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
            self.manifold, in_features=128*3, out_features=num_classes, c=self.c_out, dropout=dropout, use_bias=use_bias)
        self.prob = LorentzAct(
            self.manifold, c_in=self.c_out, c_out=self.c_out, act=act)
        self.agg = LorentzAgg(self.manifold, c=self.c_out,
                              use_att=use_att, in_features=num_classes, dropout=dropout)

    def forward(self, input):
        h = self.gin.forward(input)

        h_tangential = self.manifold.log_map_zero(h, c=self.c_out)
        h_tangential_mean = torch.mean(h_tangential.T, dim=1).T

        h_norm_tang = self.manifold.normalize_tangent_zero(
            p_tan=h_tangential_mean, c=self.c_out)
        h_classify = self.classifier(h_norm_tang)

        return h_classify, self.prob(h_classify)


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
        for i in range(num_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            if i == 0:
                block = Sequential(
                    LorentzLinear(manifold=self.manifold, in_features=in_features,
                                  out_features=128, c=c_in, dropout=dropout, use_bias=use_bias),
                    LorentzAct(manifold=self.manifold, c_in=c_in, c_out=c_out, act=ReLU()))
            else:
                block = Sequential(
                    LorentzLinear(manifold=self.manifold, in_features=128*i,
                                  out_features=128*(i+1), c=c_in, dropout=dropout, use_bias=use_bias),
                    LorentzAct(manifold=self.manifold, c_in=c_in,
                               c_out=c_out, act=ReLU())
                )

            layers.append(block)
        self.layers = Sequential(*layers)

    def forward(self, x):
        x = self.layers.forward(x)

        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
