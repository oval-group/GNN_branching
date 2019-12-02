import gurobipy as grb
import math
import torch

from itertools import product
from plnn.modules import View, Flatten
from torch import nn
from torch.nn import functional as F

class LinearizedNetwork:

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)
        # Skip all gradient computation for the weights of the Net
        for param in self.net.parameters():
            param.requires_grad = False

    def remove_maxpools(self, domain):
        from plnn.model import reluify_maxpool, simplify_network
        if any(map(lambda x: type(x) is nn.MaxPool1d, self.layers)):
            new_layers = simplify_network(reluify_maxpool(self.layers, domain))
            self.layers = new_layers


    def get_upper_bound_random(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 1024
        nb_inp = domain.shape[:-1]
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        sp_shape = (nb_samples, ) + nb_inp
        rand_samples = torch.Tensor(*sp_shape)
        rand_samples.uniform_(0, 1)

        domain_lb = domain.select(-1, 0).contiguous()
        domain_ub = domain.select(-1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.unsqueeze(0).expand(*sp_shape)
        domain_width = domain_width.unsqueeze(0).expand(*sp_shape)

        with torch.no_grad():
            inps = domain_lb + domain_width * rand_samples
            outs = self.net(inps)

            upper_bound, idx = torch.min(outs, dim=0)

            upper_bound = upper_bound[0].item()
            ub_point = inps[idx].squeeze()

        return ub_point, upper_bound

    def get_upper_bound_pgd(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 2056
        torch.set_num_threads(1)
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        best_ub = float('inf')
        best_ub_inp = None

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        inps = (domain_lb + domain_width * rand_samples)

        with torch.enable_grad():
            batch_ub = float('inf')
            for i in range(1000):
                prev_batch_best = batch_ub

                self.net.zero_grad()
                if inps.grad is not None:
                    inps.grad.zero_()
                inps = inps.detach().requires_grad_()
                out = self.net(inps)

                batch_ub = out.min().item()
                if batch_ub < best_ub:
                    best_ub = batch_ub
                    # print(f"New best lb: {best_lb}")
                    _, idx = out.min(dim=0)
                    best_ub_inp = inps[idx[0]]

                if batch_ub >= prev_batch_best:
                    break

                all_samp_sum = out.sum() / nb_samples
                all_samp_sum.backward()
                grad = inps.grad

                max_grad, _ = grad.max(dim=0)
                min_grad, _ = grad.min(dim=0)
                grad_diff = max_grad - min_grad

                lr = 1e-2 * domain_width / grad_diff
                min_lr = lr.min()

                step = -min_lr*grad
                inps = inps + step

                inps = torch.max(inps, domain_lb)
                inps = torch.min(inps, domain_ub)

        return best_ub_inp, best_ub

    get_upper_bound = get_upper_bound_random

    def get_lower_bound(self, domain, force_optim=False):
        '''
        Update the linear approximation for `domain` of the network and use it
        to compute a lower bound on the minimum of the output.

        domain: Tensor containing in each row the lower and upper bound for
                the corresponding dimension
        '''
        self.define_linear_approximation(domain, force_optim)
        return self.compute_lower_bound()

    def compute_lower_bound(self, node=(-1, None), upper_bound=False,
                            all_optim=False):
        '''
        Compute a lower bound of the function for the given node

        node: (optional) Index (as a tuple) in the list of gurobi variables of the node to optimize
              First index is the layer, second index is the neuron.
              For the second index, None is a special value that indicates to optimize all of them,
              both upper and lower bounds.
        upper_bound: (optional) Compute an upper bound instead of a lower bound
        all_optim: Should the bounds be computed only in the case where they are not already leading to
              non relaxed version. This option is only useful if the batch mode based on None in node is
              used.
        '''
        layer_with_var_to_opt = self.prerelu_gurobi_vars[node[0]]
        is_batch = (node[1] is None)
        if not is_batch:
            if isinstance(node[1], int):
                var_to_opt = layer_with_var_to_opt[node[1]]
            elif (isinstance(node[1], tuple) and isinstance(layer_with_var_to_opt, list)):
                # This is the nested list format
                to_query = layer_with_var_to_opt
                for idx in node[1]:
                    to_query = to_query[idx]
                var_to_opt = to_query
            else:
                raise NotImplementedError

            opt_direct = grb.GRB.MAXIMIZE if upper_bound else grb.GRB.MINIMIZE
            # We will make sure that the objective function is properly set up
            self.model.setObjective(var_to_opt, opt_direct)

            # We will now compute the requested lower bound
            self.model.update()
            self.model.optimize()
            assert self.model.status == 2, "LP wasn't optimally solved"

            return var_to_opt.X
        else:
            print("Batch Gurobi stuff")
            new_lbs = []
            new_ubs = []
            if isinstance(layer_with_var_to_opt, list):
                for var_idx, var in enumerate(layer_with_var_to_opt):
                    curr_lb = self.lower_bounds[node[0]][var_idx]
                    curr_ub = self.upper_bounds[node[0]][var_idx]
                    if (all_optim or
                        ((curr_lb < 0) and (curr_ub > 0))):

                        # Do the maximizing
                        self.model.setObjective(var, grb.GRB.MAXIMIZE)
                        self.model.update()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        new_ubs.append(min(curr_ub, var.X))
                        # print(f"UB was {curr_ub}, now is {new_ubs[-1]}")
                        # Do the minimizing
                        self.model.setObjective(var, grb.GRB.MINIMIZE)
                        self.model.reset()
                        self.model.update()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        new_lbs.append(max(curr_lb, var.X))
                        # print(f"LB was {curr_lb}, now is {new_lbs[-1]}")
                    else:
                        new_ubs.append(curr_ub)
                        new_lbs.append(curr_lb)
            else:
                new_lbs = self.lower_bounds[node[0]].clone()
                new_ubs = self.upper_bounds[node[0]].clone()
                bound_shape = new_lbs.shape
                for chan_idx, row_idx, col_idx in product(range(bound_shape[0]),
                                                          range(bound_shape[1]),
                                                          range(bound_shape[2])):
                    curr_lb = new_lbs[chan_idx, row_idx, col_idx]
                    curr_ub = new_ubs[chan_idx, row_idx, col_idx]
                    if (all_optim or
                        ((curr_lb < 0) and (curr_ub > 0))):
                        var = layer_with_var_to_opt[chan_idx, row_idx, col_idx]

                        # Do the maximizing
                        self.model.setObjective(var, grb.GRB.MAXIMIZE)
                        self.model.update()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        new_ubs[chan_idx, row_idx, col_idx] = min(curr_ub, var.X)
                        # print(f"UB was {curr_ub}, now is {new_ubs[chan_idx, row_idx, col_idx]}")
                        # Do the minimizing
                        self.model.setObjective(var, grb.GRB.MINIMIZE)
                        self.model.reset()
                        self.model.update()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        new_lbs[chan_idx, row_idx, col_idx] = max(curr_lb, var.X)
                        # print(f"LB was {curr_lb}, now is {new_lbs[chan_idx, row_idx, col_idx]}")

            return torch.tensor(new_lbs), torch.tensor(new_ubs)

    def build_model_using_bounds(self, input_domain, intermediate_bounds):
        self.gurobi_vars = []
        self.prerelu_gurobi_vars = []
        self.lower_bounds, self.upper_bounds = intermediate_bounds

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)


        self.zero_var = self.model.addVar(lb=0, ub=0, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'zero')
        if input_domain.dim() == 2:
            inp_gurobi_vars = self.model.addVars([i for i in range(input_domain.numel() // 2)],
                                                 lb=self.lower_bounds[0],
                                                 ub=self.upper_bounds[0],
                                                 name='inp')
            inp_gurobi_vars = [var for key, var in inp_gurobi_vars.items()]
        else:
            #inp_shape = self.lower_bounds[0].shape
            #inp_gurobi_vars = self.model.addVars([chan for chan in range(inp_shape[0])],
            #                                     [row for row in range(inp_shape[1])],
            #                                     [col for col in range(inp_shape[2])],
            #                                     lb=self.lower_bounds[0],
            #                                     ub=self.upper_bounds[0],
            #                                     name='inp')
            inp_gurobi_vars = {}
            for chan in range(input_domain.size(0)):
                chan_vars = []
                for row in range(input_domain.size(1)):
                    row_vars = []
                    for col in range(input_domain.size(2)):
                        lb = input_domain[chan, row, col, 0]
                        ub = input_domain[chan, row, col, 1]
                        v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'inp_[{chan},{row},{col}]')
                        inp_gurobi_vars[(chan, row, col)] = v
        self.gurobi_vars.append(inp_gurobi_vars)
        self.prerelu_gurobi_vars.append(inp_gurobi_vars)

        layer_idx = 1
        for layer in self.layers:
            if type(layer) is nn.Linear:
                layer_nb_out = layer.out_features
                pre_vars = self.gurobi_vars[-1]
                if isinstance(pre_vars, grb.tupledict):
                    pre_vars = [var for key, var in sorted(pre_vars.items())]
                # Build all the outputs of the linear layer
                new_vars = self.model.addVars([i for i in range(layer_nb_out)],
                                              lb=self.lower_bounds[layer_idx],
                                              ub=self.upper_bounds[layer_idx],
                                              name=f'zhat{layer_idx}')
                new_layer_gurobi_vars = [var for key, var in new_vars.items()]
                self.model.addConstrs(
                    ((grb.LinExpr(layer.weight[neuron_idx, :], pre_vars)
                     + layer.bias[neuron_idx].item()) == new_vars[neuron_idx]
                     for neuron_idx in range(layer.out_features)),
                    name=f'lay{layer_idx}'
                )
                self.prerelu_gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) is nn.Conv2d:
                print("Convolutional layer")
                in_shape = self.lower_bounds[layer_idx-1].shape
                out_shape = self.lower_bounds[layer_idx].shape

                flat_idxs = [elt for elt in product(range(out_shape[0]),
                                                    range(out_shape[1]),
                                                    range(out_shape[2]))]
                flat_out_lbs = [self.lower_bounds[layer_idx][chan, row, col]
                                for chan, row, col in product(range(out_shape[0]),
                                                              range(out_shape[1]),
                                                              range(out_shape[2]))]
                flat_out_ubs = [self.upper_bounds[layer_idx][chan, row, col]
                                for chan, row, col in product(range(out_shape[0]),
                                                              range(out_shape[1]),
                                                              range(out_shape[2]))]
                new_layer_gurobi_vars = self.model.addVars(flat_idxs,
                                                           lb=flat_out_lbs,
                                                           ub=flat_out_ubs,
                                                           name=f'zhat{layer_idx}')
                coeffs = []
                for out_chan_idx in range(out_shape[0]):
                    coeffs.append(layer.weight[out_chan_idx, :].view(-1))

                def make_lin_expr(out_chan_idx, out_row_idx, out_col_idx):
                    lin_bias = layer.bias[out_chan_idx].item()
                    lin_coeffs = coeffs[out_chan_idx]

                    start_row_idx = -layer.padding[0] + layer.stride[0]*out_row_idx
                    end_row_idx = start_row_idx + layer.weight.shape[2]
                    start_col_idx = -layer.padding[1] + layer.stride[1]*out_col_idx
                    end_col_idx = start_col_idx + layer.weight.shape[3]

                    lin_vars = [
                        (self.zero_var if ((row_idx < 0) or (row_idx == in_shape[1])
                                      or (col_idx < 0) or (col_idx == in_shape[2]))
                         else self.gurobi_vars[-1][(chan_idx, row_idx, col_idx)])
                        for chan_idx in range(in_shape[0])
                        for row_idx in range(start_row_idx, end_row_idx)
                        for col_idx in range(start_col_idx, end_col_idx)
                    ]
                    lin_expr = grb.LinExpr(lin_coeffs, lin_vars) + lin_bias
                    return lin_expr

                constrs = []
                for out_chan_idx in range(out_shape[0]):
                    for out_row_idx in range(out_shape[1]):
                        for out_col_idx in range(out_shape[2]):
                            constrs.append(make_lin_expr(out_chan_idx, out_row_idx, out_col_idx)
                                           == new_layer_gurobi_vars[(out_chan_idx, out_row_idx, out_col_idx)])
                self.model.addConstrs(constr for constr in constrs)
                self.prerelu_gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) is nn.ReLU:
                pre_lbs = self.lower_bounds[layer_idx]
                pre_ubs = self.upper_bounds[layer_idx]
                if isinstance(self.gurobi_vars[-1], grb.tupledict):
                    amb_mask = (pre_lbs < 0) & (pre_ubs>0)
                    to_new_preubs = pre_ubs[amb_mask]
                    to_new_prelbs = pre_lbs[amb_mask]

                    new_var_idxs = torch.nonzero((pre_lbs < 0) & (pre_ubs > 0)).squeeze().numpy().tolist()
                    new_var_idxs = [tuple(idxs) for idxs  in new_var_idxs]
                    new_layer_gurobi_vars = self.model.addVars(new_var_idxs,
                                                               lb=0,
                                                               ub=to_new_preubs,
                                                               name=f'z{layer_idx}')

                    flat_new_vars = [new_layer_gurobi_vars[idx] for idx in new_var_idxs]
                    pre_amb_vars = [self.gurobi_vars[-1][idx] for idx in new_var_idxs]
                    # Add the constraint that it's superior to the inputs
                    self.model.addConstrs(
                        (flat_new_vars[idx] >= pre_amb_vars[idx]
                         for idx in range(len(flat_new_vars))),
                        name=f'ReLU_lb{layer_idx}'
                    )
                    # Add the constraint that it's below the upper bound
                    slopes = to_new_preubs / (to_new_preubs - to_new_prelbs)
                    biases = -to_new_prelbs * slopes
                    self.model.addConstrs(
                        (flat_new_vars[idx] <= slopes[idx].item()*pre_amb_vars[idx] + biases[idx]
                        for idx in range(len(flat_new_vars))),
                        name=f'ReLU_ub{layer_idx}'
                    )

                    for pos in torch.nonzero(pre_lbs >= 0).squeeze().numpy().tolist():
                        pos = tuple(pos)
                        new_layer_gurobi_vars[pos] = self.gurobi_vars[-1][pos]
                    for pos in torch.nonzero(pre_ubs <= 0).squeeze().numpy().tolist():
                        new_layer_gurobi_vars[tuple(pos)] = self.zero_var
                else:
                    assert isinstance(self.gurobi_vars[-1][0], grb.Var)

                    amb_mask = (pre_lbs < 0) & (pre_ubs > 0)
                    to_new_preubs = pre_ubs[amb_mask]
                    new_var_idxs = torch.nonzero(amb_mask).squeeze().numpy().tolist()
                    new_vars = self.model.addVars(new_var_idxs,
                                                  lb=0,
                                                  ub=to_new_preubs,
                                                  name=f'z{layer_idx}')

                    # Add the constraint that it's superior to the inputs
                    self.model.addConstrs(
                        (new_vars[idx] >= self.gurobi_vars[-1][idx]
                         for idx in new_var_idxs),
                        name=f'ReLU_lb{layer_idx}'
                    )
                    slopes = pre_ubs / (pre_ubs - pre_lbs)
                    biases = -pre_lbs * slopes
                    self.model.addConstrs(
                        (new_vars[idx] <= slopes[idx].item()*self.gurobi_vars[-1][idx] + biases[idx]
                         for idx in new_var_idxs),
                        name=f'ReLU_ub{layer_idx}'
                    )
                    # Get all the variables in a list, such that we have the
                    # output of the layer
                    new_layer_gurobi_vars = []
                    new_idx = 0
                    for idx in range(layer_nb_out):
                        if pre_lbs[idx] >= 0:
                            # Pass through variable
                            new_layer_gurobi_vars.append(self.gurobi_vars[-1][idx])
                        elif pre_ubs[idx] <= 0:
                            # Blocked variable
                            new_layer_gurobi_vars.append(self.zero_var)
                        else:
                            new_layer_gurobi_vars.append(new_vars[idx])
                layer_idx += 1
            self.gurobi_vars.append(new_layer_gurobi_vars)
        self.model.update()

    def define_linear_approximation(self, input_domain, force_optim=False):
        '''
        input_domain: Tensor containing in each row the lower and upper bound
                      for the corresponding dimension
        '''
        self.lower_bounds = []
        self.upper_bounds = []
        self.gurobi_vars = []
        self.prerelu_gurobi_vars = []
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)

        ## Do the input layer, which is a special case
        inp_lbs = []
        inp_ubs = []
        inp_gurobi_vars = []
        zero_var = self.model.addVar(lb=0, ub=0, obj=0,
                                     vtype=grb.GRB.CONTINUOUS,
                                     name=f'zero')
        if input_domain.dim() == 2:
            # This is a linear input.
            for dim, (lb, ub) in enumerate(input_domain):
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.CONTINUOUS,
                                      name=f'inp_{dim}')
                inp_gurobi_vars.append(v)
                inp_lbs.append(lb)
                inp_ubs.append(ub)
        else:
            assert input_domain.dim() == 4
            for chan in range(input_domain.size(0)):
                chan_vars = []
                chan_lbs = []
                chan_ubs = []
                for row in range(input_domain.size(1)):
                    row_vars = []
                    row_lbs = []
                    row_ubs = []
                    for col in range(input_domain.size(2)):
                        lb = input_domain[chan, row, col, 0]
                        ub = input_domain[chan, row, col, 1]
                        v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'inp_[{chan},{row},{col}]')
                        row_vars.append(v)
                        row_lbs.append(lb.item())
                        row_ubs.append(ub.item())
                    chan_vars.append(row_vars)
                    chan_lbs.append(row_lbs)
                    chan_ubs.append(row_ubs)
                inp_gurobi_vars.append(chan_vars)
                inp_lbs.append(chan_lbs)
                inp_ubs.append(chan_ubs)
        self.model.update()

        self.lower_bounds.append(torch.tensor(inp_lbs))
        self.upper_bounds.append(torch.tensor(inp_ubs))
        self.gurobi_vars.append(inp_gurobi_vars)
        self.prerelu_gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        for layer in self.layers:
            is_final = (layer is self.layers[-1])
            new_layer_lb = []
            new_layer_ub = []
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                pre_lb = self.lower_bounds[-1]
                pre_ub = self.upper_bounds[-1]
                pre_vars = self.gurobi_vars[-1]
                if pre_lb.dim() > 1:
                    pre_lb = pre_lb.view(-1)
                    pre_ub = pre_ub.view(-1)
                    pre_vars = []
                    for chan_idx in range(len(self.gurobi_vars[-1])):
                        for row_idx in range(len(self.gurobi_vars[-1][chan_idx])):
                            pre_vars.extend(self.gurobi_vars[-1][chan_idx][row_idx])
                if layer_idx > 1:
                    # The previous bounds are from a ReLU
                    pre_lb = torch.clamp(pre_lb, 0, None)
                    pre_ub = torch.clamp(pre_ub, 0, None)
                pos_w = torch.clamp(layer.weight, 0, None)
                neg_w = torch.clamp(layer.weight, None, 0)
                out_lbs = pos_w @ pre_lb + neg_w @ pre_ub + layer.bias
                out_ubs = pos_w @ pre_ub + neg_w @ pre_lb + layer.bias

                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, pre_vars)

                    out_lb = out_lbs[neuron_idx].item()
                    out_ub = out_ubs[neuron_idx].item()
                    v = self.model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                                          obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    should_opt = (force_optim
                                  or is_final
                                  or ((layer_idx > 1) and (out_lb < 0) and (out_ub > 0))
                    )
                    if should_opt:
                        self.model.setObjective(v, grb.GRB.MINIMIZE)
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        # We have computed a lower bound
                        out_lb = v.X

                        # Let's now compute an upper bound
                        self.model.setObjective(v, grb.GRB.MAXIMIZE)
                        self.model.update()
                        self.model.reset()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        out_ub = v.X

                    new_layer_lb.append(out_lb)
                    new_layer_ub.append(out_ub)
                    new_layer_gurobi_vars.append(v)
                self.lower_bounds.append(torch.tensor(new_layer_lb))
                self.upper_bounds.append(torch.tensor(new_layer_ub))
                self.prerelu_gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)
                pre_lb = self.lower_bounds[-1].unsqueeze(0)
                pre_ub = self.upper_bounds[-1].unsqueeze(0)
                if layer_idx > 1:
                    # The previous bounds are from a ReLU
                    pre_lb = torch.clamp(pre_lb, 0, None)
                    pre_ub = torch.clamp(pre_ub, 0, None)
                pos_weight = torch.clamp(layer.weight, 0, None)
                neg_weight = torch.clamp(layer.weight, None, 0)

                out_lbs = (F.conv2d(pre_lb, pos_weight, layer.bias,
                                    layer.stride, layer.padding, layer.dilation, layer.groups)
                           + F.conv2d(pre_ub, neg_weight, None,
                                      layer.stride, layer.padding, layer.dilation, layer.groups))
                out_ubs = (F.conv2d(pre_ub, pos_weight, layer.bias,
                                    layer.stride, layer.padding, layer.dilation, layer.groups)
                           + F.conv2d(pre_lb, neg_weight, None,
                                      layer.stride, layer.padding, layer.dilation, layer.groups))
                for out_chan_idx in range(out_lbs.size(1)):
                    out_chan_lbs = []
                    out_chan_ubs = []
                    out_chan_vars = []
                    for out_row_idx in range(out_lbs.size(2)):
                        out_row_lbs = []
                        out_row_ubs = []
                        out_row_vars = []
                        for out_col_idx in range(out_lbs.size(3)):
                            lin_expr = layer.bias[out_chan_idx].item()

                            for in_chan_idx in range(layer.weight.shape[1]):
                                for ker_row_idx in range(layer.weight.shape[2]):
                                    in_row_idx = -layer.padding[0] + layer.stride[0]*out_row_idx + ker_row_idx
                                    if (in_row_idx < 0) or (in_row_idx == pre_lb.size(2)):
                                        # This is padding -> value of 0
                                        continue
                                    for ker_col_idx in range(layer.weight.shape[3]):
                                        in_col_idx = -layer.padding[1] + layer.stride[1]*out_col_idx + ker_col_idx
                                        if (in_col_idx < 0) or (in_col_idx == pre_lb.size(3)):
                                            # This is padding -> value of 0
                                            continue
                                        coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                                        if abs(coeff) > 1e-6:
                                            lin_expr += coeff * self.gurobi_vars[-1][in_chan_idx][in_row_idx][in_col_idx]

                            out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()

                            v = self.model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(v == lin_expr)
                            self.model.update()

                            should_opt = (force_optim
                                          or is_final
                                          or ((layer_idx > 1) and (out_lb < 0) and (out_ub > 0))
                                          )
                            if should_opt:
                                self.model.setObjective(v, grb.GRB.MINIMIZE)
                                self.model.reset()
                                self.model.optimize()
                                assert self.model.status == 2, "LP wasn't optimally solved"
                                # We have computed a lower bound
                                out_lb = v.X

                                # Let's now compute an upper bound
                                self.model.setObjective(v, grb.GRB.MAXIMIZE)
                                self.model.update()
                                self.model.reset()
                                self.model.optimize()
                                assert self.model.status == 2, "LP wasn't optimally solved"
                                out_ub = v.X

                            out_row_vars.append(v)
                            out_row_lbs.append(out_lb)
                            out_row_ubs.append(out_ub)
                        out_chan_vars.append(out_row_vars)
                        out_chan_lbs.append(out_row_lbs)
                        out_chan_ubs.append(out_row_ubs)
                    new_layer_gurobi_vars.append(out_chan_vars)
                    new_layer_lb.append(out_chan_lbs)
                    new_layer_ub.append(out_chan_ubs)
                self.lower_bounds.append(torch.tensor(new_layer_lb))
                self.upper_bounds.append(torch.tensor(new_layer_ub))
                self.prerelu_gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) == nn.ReLU:
                if isinstance(self.gurobi_vars[-1][0], list):
                    # This is convolutional
                    pre_lbs = torch.Tensor(self.lower_bounds[-1])
                    pre_ubs = torch.Tensor(self.upper_bounds[-1])
                    for chan_idx, channel in enumerate(self.gurobi_vars[-1]):
                        chan_vars = []
                        chan_lbs = []
                        chan_ubs = []
                        for row_idx, row in enumerate(channel):
                            row_vars = []
                            row_lbs = []
                            row_ubs = []
                            for col_idx, pre_var in enumerate(row):
                                pre_lb = pre_lbs[chan_idx, row_idx, col_idx].item()
                                pre_ub = pre_ubs[chan_idx, row_idx, col_idx].item()

                                if pre_lb >= 0 and pre_ub >= 0:
                                    # ReLU is always passing
                                    lb = pre_lb
                                    ub = pre_ub
                                    v = pre_var
                                elif pre_lb <= 0 and pre_ub <= 0:
                                    lb = 0
                                    ub = 0
                                    v = zero_var
                                else:
                                    lb = 0
                                    ub = pre_ub
                                    v = self.model.addVar(lb=lb, ub=ub,
                                                          obj=0, vtype=grb.GRB.CONTINUOUS,
                                                          name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                    self.model.addConstr(v >= pre_var)
                                    slope = pre_ub / (pre_ub - pre_lb)
                                    bias = - pre_lb * slope
                                    self.model.addConstr(v <= slope*pre_var + bias)
                                row_vars.append(v)
                            chan_vars.append(row_vars)
                        new_layer_gurobi_vars.append(chan_vars)
                else:
                    assert isinstance(self.gurobi_vars[-1][0], grb.Var)
                    for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
                        pre_lb = self.lower_bounds[-1][neuron_idx]
                        pre_ub = self.upper_bounds[-1][neuron_idx]

                        v = self.model.addVar(lb=max(0, pre_lb),
                                              ub=max(0, pre_ub),
                                              obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'ReLU{layer_idx}_{neuron_idx}')
                        if pre_lb >= 0 and pre_ub >= 0:
                            # The ReLU is always passing
                            self.model.addConstr(v == pre_var)
                            lb = pre_lb
                            ub = pre_ub
                        elif pre_lb <= 0 and pre_ub <= 0:
                            lb = 0
                            ub = 0
                            # No need to add an additional constraint that v==0
                            # because this will be covered by the bounds we set on
                            # the value of v.
                        else:
                            lb = 0
                            ub = pre_ub
                            self.model.addConstr(v >= pre_var)

                            slope = pre_ub / (pre_ub - pre_lb)
                            bias = - pre_lb * slope
                            self.model.addConstr(v <= slope.item() * pre_var + bias.item())

                        new_layer_gurobi_vars.append(v)
            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                continue
            else:
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

        self.model.update()
