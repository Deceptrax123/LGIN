from layers.lorentz_ops import LorentzAct, LorentzLinear, LorentzAgg
from torch.nn import Module
from torch import Tensor


class LorentzGIN(Module):
    def __init__(self, manifold, eps, in_features, c_in, nn, use_bias, use_att):
        super(LorentzGIN, self).__init__()

        self.c_in = c_in
        self.manifold = manifold
        self.agg = LorentzAgg(manifold, self.c_in,
                              use_att, in_features, use_bias)
        self.nn = nn
        self.eps = eps

    def forward(self, input):
        x, adj = input

        # Map to hyperbolid space
        x = self.manifold.exp_map_zero(x, self.c_in)
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.agg.forward(x=x, adj=adj)  # Calculate the Frechet Mean

        x_r = x[1]
        if x_r is not None:
            out = self.manifold.exp_map_zero(self.manifold.log_map_zero(
                out, self.c_in)+(1+self.eps)*self.manifold.log_map_zero(x_r, self.c_in))

        return self.nn(out)


class LorentzGNN(Module):
    def __init__(self, manifold, in_features, out_features, c_in, c_out, drop_out, act, use_bias, use_att):
        super(LorentzGNN, self).__init__()

        self.c_in = c_in
        self.linear = LorentzLinear(
            manifold, in_features, out_features, c_in, drop_out, use_bias)
        self.agg = LorentzAgg(manifold, c_in, use_att, out_features, drop_out)
        self.lorentz_act = LorentzAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.lorentz_act.forward(h)

        output = h, adj

        return output

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.agg.reset_parameters()
