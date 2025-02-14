from layers.hyp import LorentzAct, LorentzLinear, LorentzAgg
from torch.nn import Module


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
