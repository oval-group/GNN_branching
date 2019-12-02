import gurobipy as grb
import torch

from itertools import product
from torch import nn
from plnn.modules import View, Flatten
from torch.nn import functional as F
from plnn.dual_network_linear_approximation import LooseDualNetworkApproximation
from plnn.network_linear_approximation import LinearizedNetwork

class MIPNetwork:

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

        # Initialize a LinearizedNetwork object to determine the lower and
        # upper bounds at each layer.
        self.lin_net = LinearizedNetwork(layers)

    def solve(self, inp_domain, timeout=None):
        '''
        inp_domain: Tensor containing in each row the lower and upper bound
                    for the corresponding dimension

        Returns:
        sat     : boolean indicating whether the MIP is satisfiable.
        solution: Feasible point if the MIP is satisfiable,
                  None otherwise.
        timeout : Maximum allowed time to run, if is not None
        '''
        if self.lower_bounds[-1].min() > 0:
            print("Early stopping")
            # The problem is infeasible, and we haven't setup the MIP
            return (False, None, 0)

        if timeout is not None:
            self.model.setParam('TimeLimit', timeout)

        if self.check_obj_value_callback:
            def early_stop_cb(model, where):
                if where == grb.GRB.Callback.MIP:
                    best_bound = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
                    if best_bound > 0:
                        model.terminate()

                if where == grb.GRB.Callback.MIPNODE:
                    nodeCount = model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT)
                    if (nodeCount % 100) == 0:
                        print(f"Running Nb states visited: {nodeCount}")

                if where == grb.GRB.Callback.MIPSOL:
                    obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                    if obj < 0:
                        # Does it have a chance at being a valid
                        # counter-example?

                        # Check it with the network
                        input_vals = model.cbGetSolution(self.gurobi_vars[0])

                        with torch.no_grad():
                            if isinstance(input_vals, list):
                                inps = torch.Tensor(input_vals).view(1, -1)
                            else:
                                assert isinstance(input_vals, grb.tupledict)
                                inps = torch.Tensor([val for val in input_vals.values()])
                                inps = inps.view((1,) + self.lower_bounds[0].shape)
                            out = self.net(inps).squeeze()
                            # In case there is several output to the network, get the minimum one.
                            out = out.min().item()

                        if out < 0:
                            model.terminate()
        else:
            def early_stop_cb(model, where):
                if where == grb.GRB.Callback.MIPNODE:
                    nodeCount = model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT)
                    if (nodeCount % 100) == 0:
                        print(f"Running Nb states visited: {nodeCount}")

        self.model.optimize(early_stop_cb)
        nb_visited_states = self.model.nodeCount

        if self.model.status is grb.GRB.INFEASIBLE:
            # Infeasible: No solution
            return (False, None, nb_visited_states)
        elif self.model.status is grb.GRB.OPTIMAL:
            # There is a feasible solution. Return the feasible solution as well.
            len_inp = len(self.gurobi_vars[0])

            # Get the input that gives the feasible solution.
            #input_vals = model.cbGetSolution(self.gurobi_vars[0])
            #inps = torch.Tensor([val for val in input_vals.values()])
            #inps = inps.view((1,) + self.lower_bounds[0].shape)
            optim_val = self.gurobi_vars[-1][-1].x

            return (optim_val < 0, (None, optim_val), nb_visited_states)
        elif self.model.status is grb.GRB.INTERRUPTED:
            obj_bound = self.model.ObjBound

            if obj_bound > 0:
                return (False, None, nb_visited_states)
            else:
                # There is a feasible solution. Return the feasible solution as well.
                len_inp = len(self.gurobi_vars[0])

                # Get the input that gives the feasible solution.
                inp = torch.Tensor(len_inp)
                if isinstance(self.gurobi_vars[0], list):
                    for idx, var in enumerate(self.gurobi_vars[0]):
                        inp[idx] = var.x
                else:
                    #assert isinstance(self.gurobi_vars[0], grb.tupledict)
                    inp = torch.zeros_like(self.lower_bounds[0])
                    for idx, var in self.gurobi_vars[0].items():
                        inp[idx] = var.x
                optim_val = self.gurobi_vars[-1][-1].x
            return (optim_val < 0, (inp, optim_val), nb_visited_states)
        elif self.model.status is grb.GRB.TIME_LIMIT:
            # We timed out, return a None Status
            return (None, None, nb_visited_states)
        else:
            raise Exception("Unexpected Status code")

    def tune(self, param_outfile, tune_timeout):
        self.model.Params.tuneOutput = 1
        self.model.Params.tuneTimeLimit = tune_timeout
        self.model.tune()

        # Get the best set of parameters
        self.model.getTuneResult(0)

        self.model.write(param_outfile)

    def do_interval_analysis(self, inp_domain):
        self.lower_bounds = []
        self.upper_bounds = []

        self.lower_bounds.append(inp_domain.select(-1, 0))
        self.upper_bounds.append(inp_domain.select(-1, 1))
        layer_idx = 1
        current_lb = self.lower_bounds[-1]
        current_ub = self.upper_bounds[-1]
        for layer in self.layers:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                if type(layer) is nn.Linear:
                    pos_weights = torch.clamp(layer.weight, min=0)
                    neg_weights = torch.clamp(layer.weight, max=0)

                    new_layer_lb = torch.mv(pos_weights, current_lb) + \
                                   torch.mv(neg_weights, current_ub) + \
                                   layer.bias
                    new_layer_ub = torch.mv(pos_weights, current_ub) + \
                                   torch.mv(neg_weights, current_lb) + \
                                   layer.bias
                elif type(layer) is nn.Conv2d:
                    pre_lb = torch.Tensor(current_lb).unsqueeze(0)
                    pre_ub = torch.Tensor(current_ub).unsqueeze(0)
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
                    new_layer_lb = out_lbs.squeeze(0)
                    new_layer_ub = out_ubs.squeeze(0)
                self.lower_bounds.append(new_layer_lb)
                self.upper_bounds.append(new_layer_ub)
                current_lb = new_layer_lb
                current_ub = new_layer_ub
            elif type(layer) == nn.ReLU:
                current_lb = torch.clamp(current_lb, min=0)
                current_ub = torch.clamp(current_ub, min=0)
            elif type(layer) == nn.MaxPool1d:
                new_layer_lb = []
                new_layer_ub = []
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported Maxpool option"

                nb_pre = len(self.lower_bounds[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size

                while pre_window_end <= nb_pre:
                    lb = max(current_lb[pre_start_idx:pre_window_end])
                    ub = max(current_ub[pre_start_idx:pre_window_end])

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)

                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size
                current_lb = torch.Tensor(new_layer_lb)
                current_ub = torch.Tensor(new_layer_ub)
                self.lower_bounds.append(current_lb)
                self.upper_bounds.append(current_ub)
            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                current_lb = current_lb.view(-1)
                current_ub = current_ub.view(-1)
            else:
                raise NotImplementedError


    def setup_model(self, inp_domain,
                    use_obj_function=False,
                    bounds="opt",
                    parameter_file=None):
        '''
        inp_domain: Tensor containing in each row the lower and upper bound
                    for the corresponding dimension

        optimal: If False, don't use any objective function, simply add a constraint on the output
                 If True, perform optimization and use callback to interrupt the solving when a
                          counterexample is found
        bounds: string, indicate what type of method should be used to get the intermediate bounds
        parameter_file: Load a set of parameters for the MIP solver if a path is given.

        Setup the model to be optimized by Gurobi
        '''
        if bounds == "opt":
            # First use define_linear_approximation from LinearizedNetwork to
            # compute upper and lower bounds to be able to define Ms
            self.lin_net.define_linear_approximation(inp_domain)

            self.lower_bounds = list(map(torch.Tensor, self.lin_net.lower_bounds))
            self.upper_bounds = list(map(torch.Tensor, self.lin_net.upper_bounds))
        elif bounds == "interval":
            self.do_interval_analysis(inp_domain)
            if self.lower_bounds[-1][0] > 0:
                # The problem is already guaranteed to be infeasible,
                # Let's not waste time setting up the MIP
                return
        elif bounds == "interval-kw":
            self.do_interval_analysis(inp_domain)
            kw_dual = LooseDualNetworkApproximation(self.layers)
            kw_dual.remove_maxpools(inp_domain, no_opt=True)
            lower_bounds, upper_bounds = kw_dual.get_intermediate_bounds(inp_domain)
            #print(lower_bounds)
            #print(upper_bounds)

            # We want to get the best out of interval-analysis and K&W

            # TODO: There is a slight problem. To use the K&W code directly, we
            # need to make a bunch of changes, notably remove all of the
            # Maxpooling and convert them to ReLUs. Quick and temporary fix:
            # take the max of both things if the shapes are all the same so
            # far, and use the one from interval analysis after the first
            # difference.

            # If the network are full ReLU, there should be no problem.
            # If the network are just full ReLU with a MaxPool at the end,
            # that's still okay because we get the best bounds until the
            # maxpool, and that's the last thing that we use the bounds for
            # This is just going to suck if we have a Maxpool early in the
            # network, and even then, that just means we use interval analysis
            # so stop complaining.
            for i in range(len(lower_bounds)):
                if lower_bounds[i].shape == self.lower_bounds[i].shape:
                    # Keep the best lower bound
                    lb_diff = lower_bounds[i] - self.lower_bounds[i]
                    ub_diff = upper_bounds[i] - self.upper_bounds[i]
                    # print(f"LB Difference (kw to interval) min: {lb_diff.min()} \t max:{lb_diff.max()}")
                    # print(f"UB Difference (kw to interval) min: {ub_diff.min()} \t max:{ub_diff.max()}")
                    torch.max(lower_bounds[i], self.lower_bounds[i], out=self.lower_bounds[i])
                    torch.min(upper_bounds[i], self.upper_bounds[i], out=self.upper_bounds[i])
                else:
                    # Mismatch in dimension.
                    # Drop it and stop trying to improve the stuff of interval analysis
                    break
            if self.lower_bounds[-1].min() > 0:
                # The problem is already guaranteed to be infeasible,
                # Let's not waste time setting up the MIP
                return
        else:
            raise NotImplementedError("Unknown bound computation method.")

        self.gurobi_vars = []
        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)
        self.model.setParam('DualReductions', 0)
        if parameter_file is not None:
            self.model.read(parameter_file)

        self.zero_var = self.model.addVar(lb=0, ub=0, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'zero')

        # First add the input variables as Gurobi variables.
        if inp_domain.dim() == 2:
            inp_gurobi_vars = self.model.addVars([i for i in range(inp_domain.numel() // 2)],
                                                 lb=self.lower_bounds[0],
                                                 ub=self.upper_bounds[0],
                                                 name='inp')
            inp_gurobi_vars = [var for key, var in inp_gurobi_vars.items()]
        else:
            inp_shape = self.lower_bounds[0].shape
            #inp_gurobi_vars = self.model.addVars([chan for chan in range(inp_shape[0])],
            #                                     [row for row in range(inp_shape[1])],
            #                                     [col for col in range(inp_shape[2])],
            #                                     lb=self.lower_bounds[0].numpy(),
            #                                     ub=self.upper_bounds[0].numpy(),
            #                                     name='inp')
            #import pdb; pdb.set_trace()
            inp_gurobi_vars = {}
            for chan in range(inp_domain.size(0)):
                chan_vars = []
                for row in range(inp_domain.size(1)):
                    row_vars = []
                    for col in range(inp_domain.size(2)):
                        lb = inp_domain[chan, row, col, 0]
                        ub = inp_domain[chan, row, col, 1]
                        v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'inp_[{chan},{row},{col}]')
                        inp_gurobi_vars[(chan, row, col)] = v
        self.gurobi_vars.append(inp_gurobi_vars)

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
            elif type(layer) is nn.Conv2d:
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
            elif type(layer) == nn.ReLU:
                pre_lbs = self.lower_bounds[layer_idx]
                pre_ubs = self.upper_bounds[layer_idx]
                if isinstance(self.gurobi_vars[-1], grb.tupledict):
                    amb_mask = (pre_lbs < 0) & (pre_ubs>0)
                    if amb_mask.sum().item() != 0:
                        to_new_preubs = pre_ubs[amb_mask]
                        to_new_prelbs = pre_lbs[amb_mask]

                        new_var_idxs = torch.nonzero((pre_lbs < 0) & (pre_ubs > 0)).numpy().tolist()
                        new_var_idxs = [tuple(idxs) for idxs  in new_var_idxs]
                        new_layer_gurobi_vars = self.model.addVars(new_var_idxs,
                                                                   lb=0,
                                                                   ub=to_new_preubs,
                                                                   name=f'z{layer_idx}')
                        new_binary_vars = self.model.addVars(new_var_idxs,
                                                             lb=0, ub=1,
                                                             vtype=grb.GRB.BINARY,
                                                             name=f'delta{layer_idx}')

                        flat_new_vars = [new_layer_gurobi_vars[idx] for idx in new_var_idxs]
                        flat_binary_vars = [new_binary_vars[idx] for idx in new_var_idxs]
                        pre_amb_vars = [self.gurobi_vars[-1][idx] for idx in new_var_idxs]

                        # C1: Superior to 0
                        # C2: Add the constraint that it's superior to the inputs
                        self.model.addConstrs(
                            (flat_new_vars[idx] >= pre_amb_vars[idx]
                             for idx in range(len(flat_new_vars))),
                            name=f'ReLU_lb{layer_idx}'
                        )
                        # C3: Below binary*upper_bound
                        self.model.addConstrs(
                            (flat_new_vars[idx] <= to_new_preubs[idx].item() * flat_binary_vars[idx]
                             for idx in range(len(flat_new_vars))),
                            name=f'ReLU{layer_idx}_ub1-'
                        )
                        # C4: Below binary*lower_bound
                        self.model.addConstrs(
                            (flat_new_vars[idx] <= (pre_amb_vars[idx]
                                                    - to_new_prelbs[idx].item() * (1 - flat_binary_vars[idx]))
                            for idx in range(len(flat_new_vars))),
                            name=f'ReLU{layer_idx}_ub2-'
                        )
                    else:
                        new_layer_gurobi_vars = grb.tupledict()

                    for pos in torch.nonzero(pre_lbs >= 0).numpy().tolist():
                        pos = tuple(pos)
                        new_layer_gurobi_vars[pos] = self.gurobi_vars[-1][pos]
                    for pos in torch.nonzero(pre_ubs <= 0).numpy().tolist():
                        new_layer_gurobi_vars[tuple(pos)] = self.zero_var
                else:
                    assert isinstance(self.gurobi_vars[-1][0], grb.Var)

                    amb_mask = (pre_lbs < 0) & (pre_ubs > 0)
                    if amb_mask.sum().item() == 0:
                        pass
                        # print("WARNING: No ambiguous ReLU at a layer")
                    else:
                        to_new_preubs = pre_ubs[amb_mask]
                        new_var_idxs = torch.nonzero(amb_mask).squeeze(1).numpy().tolist()
                        new_vars = self.model.addVars(new_var_idxs,
                                                      lb=0,
                                                      ub=to_new_preubs,
                                                      name=f'z{layer_idx}')
                        new_binary_vars = self.model.addVars(new_var_idxs,
                                                             lb=0, ub=1,
                                                             vtype=grb.GRB.BINARY,
                                                             name=f'delta{layer_idx}')

                        # C1: Superior to 0
                        # C2: Add the constraint that it's superior to the inputs
                        self.model.addConstrs(
                            (new_vars[idx] >= self.gurobi_vars[-1][idx]
                             for idx in new_var_idxs),
                            name=f'ReLU_lb{layer_idx}'
                        )
                        # C3: Below binary*upper_bound
                        self.model.addConstrs(
                            (new_vars[idx] <= pre_ubs[idx].item() * new_binary_vars[idx]
                             for idx in new_var_idxs),
                            name=f'ReLU{layer_idx}_ub1-'
                        )
                        # C4: Below binary*lower_bound
                        self.model.addConstrs(
                            (new_vars[idx] <= (self.gurobi_vars[-1][idx]
                                               - pre_lbs[idx].item() * (1 - new_binary_vars[idx]))
                            for idx in new_var_idxs),
                            name=f'ReLU{layer_idx}_ub2-'
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
            elif type(layer) == nn.MaxPool1d:
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported MaxPool option"
                nb_pre = len(self.gurobi_vars[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size

                while pre_window_end <= nb_pre:
                    ub_max = max(self.upper_bounds[layer_idx-1][pre_start_idx:pre_window_end]).item()
                    window_bin_vars = []
                    neuron_idx = pre_start_idx % stride
                    v = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=-grb.GRB.INFINITY,
                                          ub=grb.GRB.INFINITY,
                                          name=f'MaxPool_out_{layer_idx}_{neuron_idx}')
                    for pre_var_idx, pre_var in enumerate(self.gurobi_vars[-1][pre_start_idx:pre_window_end]):
                        lb = self.lower_bounds[layer_idx-1][pre_start_idx + pre_var_idx].item()
                        b = self.model.addVar(vtype=grb.GRB.BINARY,
                                              name= f'MaxPool_b_{layer_idx}_{neuron_idx}_{pre_var_idx}')
                        # MIP formulation of max pooling:
                        #
                        # y = max(x_1, x_2, ..., x_n)
                        #
                        # Introduce binary variables d_1, d_2, ..., d_n:
                        # d_i = i if x_i is the maximum value, 0 otherwise
                        #
                        # We know the lower (l_i) and upper bounds (u_i) for x_i
                        #
                        # Denote the maximum of the upper_bounds of all inputs x_i as u_max
                        #
                        # MIP must then satisfy the following constraints:
                        #
                        # Constr_1: l_i <= x_i <= u_i
                        # Constr_2: y >= x_i
                        # Constr_3: y <= x_i + (u_max - l_i)*(1 - d_i)
                        # Constr_4: sum(d_1, d_2, ..., d_n) yer= 1

                        # Constr_1 is already satisfied due to the implementation of LinearizedNetworks.
                        # Constr_2
                        self.model.addConstr(v >= pre_var)
                        # Constr_3
                        self.model.addConstr(v <= pre_var + (ub_max - lb)*(1-b))

                        window_bin_vars.append(b)
                    # Constr_4
                    self.model.addConstr(sum(window_bin_vars) == 1)
                    self.model.update()
                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size
                    new_layer_gurobi_vars.append(v)
            elif isinstance(layer, View) or isinstance(layer, Flatten):
                continue
            else:
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)

        if len(self.gurobi_vars[-1]) == 1:
            # The network has a scalar output, it works like this.
            pass
        else:
            # The network has multiple outputs, we need to encode that the
            # minimum is below 0, let's add a variable here that corresponds to
            # the minimum
            min_var = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                        lb=self.lower_bounds[-1].min().item(),
                                        ub=self.upper_bounds[-1].min().item(),
                                        name="final_output")
            self.model.addConstrs(
                (min_var <= self.gurobi_vars[-1][out_idx]
                for out_idx in range(len(self.gurobi_vars[-1]))),
                name=f'final_constraint_min_ub'
            )

            bin_min_vars = self.model.addVars(range(len(self.gurobi_vars[-1])),
                                              vtype=grb.GRB.BINARY,
                                              lb=0, ub=1,
                                              name='final_binary')
            out_lbmin = self.lower_bounds[-1].min()
            self.model.addConstrs(
                (min_var >= (self.gurobi_vars[-1][out_idx]
                            + (out_lbmin - self.upper_bounds[-1][out_idx]).item() * (1 - bin_min_vars[out_idx]))
                for out_idx in range(len(self.gurobi_vars[-1]))),
                name=f'final_constraint_min_lb'
            )
            self.model.addConstr(sum(var for var in bin_min_vars.values()) == 1)

            self.gurobi_vars.append([min_var])
            self.lower_bounds.append(self.lower_bounds[-1].min())
            self.upper_bounds.append(self.upper_bounds[-1].min())

        # Add the final constraint that the output must be less than or equal
        # to zero.
        if not use_obj_function:
            self.model.addConstr(self.gurobi_vars[-1][0] <= 0)
            self.model.setObjective(0, grb.GRB.MAXIMIZE)
            self.check_obj_value_callback = False
        else:
            # Set the minimization of the network output
            self.model.setObjective(self.gurobi_vars[-1][-1], grb.GRB.MINIMIZE)
            self.check_obj_value_callback = True

        # Optimize the model.
        self.model.update()
        #self.model.write('new_debug.lp')
