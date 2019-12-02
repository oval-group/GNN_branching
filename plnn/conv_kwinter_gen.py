import gurobipy as grb
import math
import torch
from convex_adversarial import DualNetwork
from convex_adversarial.dual_layers import DualLinear, DualReLU
#from plnn.dual_network_mem import LooseDualNetworkApproximation
from plnn.dual_network_linear_approximation import LooseDualNetworkApproximation
from plnn.modules import View, Flatten
from torch.autograd import Variable
from torch.nn import functional as F
import time
from torch import nn

class LinearizedNetwork:

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

    def get_upper_bound_random(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 2056
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)
        with torch.no_grad():

            inps = domain_lb + domain_width * rand_samples
            outs = self.net(inps)

            upper_bound, idx = torch.min(outs, dim=0)

            upper_bound = upper_bound[0].item()
            ub_point = inps[idx].squeeze()
        return ub_point, upper_bound

    def get_upper_bound_pgd(self, domain_lb, domain_ub, ub_point):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 2056
        torch.set_num_threads(1)
        nb_inp = torch.tensor(ub_point.size()).type(torch.long)
        nb_inp[0] = nb_samples
        nb_inp = nb_inp.tolist()

        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        #rand_samples = torch.randn(nb_inp)
        rand_samples = torch.rand(nb_inp)

        best_ub = float('inf')
        best_ub_inp = None

        #domain_lb = domain.select(1, 0).contiguous()
        #domain_ub = domain.select(1, 1).contiguous()
        domain_lb = domain_lb.unsqueeze(0)
        domain_ub = domain_ub.unsqueeze(0)
        domain_width = domain_ub - domain_lb

        ub_point_expanded = ub_point.expand(nb_inp)
        domain_width = domain_width.expand(nb_inp)

        domain_lb = domain_lb.expand(nb_inp)
        inps = domain_lb + domain_width * rand_samples

        #inps = ub_point_expanded + (domain_width/2) * rand_samples
        #inps = torch.max(domain_lb, inps)
        #inps = torch.min(domain_ub, inps)
        inps[0] = ub_point.clone()
        inps = Variable(inps, requires_grad=True)


        batch_ub = float('inf')
        for i in range(1000):
            prev_batch_best = batch_ub

            self.net.zero_grad()
            #inps.zero_grad()

            out = self.net(inps)

            batch_ub = out.min().item()
            if batch_ub < best_ub:
                best_ub = batch_ub
                # print(f"New best lb: {best_lb}")
                _, idx = out.min(dim=0)
                best_ub_inp = inps[idx[0]]

            if batch_ub >= prev_batch_best:
                break
            #print(best_ub)
            all_samp_sum = out.sum() / nb_samples
            all_samp_sum.backward()
            grad = inps.grad

            max_grad, _ = grad.max(dim=0)
            min_grad, _ = grad.min(dim=0)
            grad_diff = max_grad - min_grad

            lr = 1e-2 * domain_width / grad_diff
            min_lr = lr.min()

            with torch.no_grad():
                step = -min_lr*grad
                inps += step
                #inps= inps.clamp(domain_lb, domain_ub)

                inps = torch.max(domain_lb,inps)
                inps = torch.min(inps, domain_ub)
                inps = Variable(inps, requires_grad=True)

        return best_ub_inp, best_ub

    get_upper_bound = get_upper_bound_pgd

class InfeasibleMaskException(Exception):
    pass

class KWConvGen(LinearizedNetwork):

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        super(KWConvGen, self).__init__(layers)


    def get_lower_bound(self, relu_mask,  pre_lbs, pre_ubs, decision, choice):
        try:
            start = time.time()
            gub, glb, ub_point, dual_vars, dual_vars_other, primals, new_mask, lower_bounds, upper_bounds = self.update_the_model(relu_mask, pre_lbs, pre_ubs, decision, choice)
            end = time.time()
            print('KW_Int define_linear: ', end-start)
            return gub, glb,ub_point, dual_vars, dual_vars_other, primals,new_mask, lower_bounds, upper_bounds
        except InfeasibleMaskException:
            # The model is infeasible, so this mask is wrong.
            # We just return an infinite lower bound
            return float('inf'), float('inf'), None,None, None, None,relu_mask, None, None


    def check_optimization_success(self, introduced_constrs_all=None):
        if self.model.status == 2:
            # Optimization successful, nothing to complain about
            pass
        elif self.model.status == 3:
            for introduced_cons_layer in introduced_constrs_all:
                self.model.remove(introduced_cons_layer)
            # The model is infeasible. We have made incompatible
            # assumptions, so this subdomain doesn't exist.
            raise InfeasibleMaskException()
        else:
            print('\n')
            print(f'model.status: {self.model.status}\n')
            #import pdb; pdb.set_trace()
            raise NotImplementedError


    def build_the_model(self, input_domain, x, ball_eps, bounded):
        '''
        Before the first branching, we build the model and create a mask matrix
        
        Output: relu_mask, current intermediate upper and lower bounds, a list of 
                indices of the layers right before a Relu layer
                the constructed gurobi model
        
        NOTE: we keep all bounds as a list of tensors from now on.
              Only lower and upper bounds are kept in the same shape as layers' outputs.
              Mask is linearized 
              Gurobi_var lists are lineariezd
              self.model_lower_bounds and self.model_upper_bounds are kepts mainly for
              debugging purpose and could be removed

        '''
        new_relu_mask = []
        lower_bounds = []
        upper_bounds = []
        
        ## NEW STRUCTURE: deal with all available bounds first
        # first get KW bounds
        self.loose_dual = LooseDualNetworkApproximation(self.layers, x, ball_eps)
        kw_lb, kw_ub, pre_relu_indices, dual_info = self.loose_dual.init_kw_bounds(bounded)
        # second get interval bounds
        if len(input_domain.size()) == 2:
            lower_bounds.append(input_domain[:,0].squeeze(-1))
            upper_bounds.append(input_domain[:,1].squeeze(-1))
        else:
            lower_bounds.append(input_domain[:,:,:,0].squeeze(-1))
            upper_bounds.append(input_domain[:,:,:,1].squeeze(-1))
        layer_idx = 1
        for layer in self.layers:
            new_layer_lb = []
            new_layer_ub = []
            if type(layer) is nn.Linear:
                pre_lb = lower_bounds[-1]
                pre_ub = upper_bounds[-1]
                pos_w = torch.clamp(layer.weight, 0, None)
                neg_w = torch.clamp(layer.weight, None, 0)
                out_lbs = pos_w @ pre_lb + neg_w @ pre_ub + layer.bias
                out_ubs = pos_w @ pre_ub + neg_w @ pre_lb + layer.bias
                # Get the better estimates from KW and Interval Bounds
                new_layer_lb = torch.max(kw_lb[layer_idx], out_lbs)
                new_layer_ub = torch.min(kw_ub[layer_idx], out_ubs)
            elif type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)
                pre_lb = lower_bounds[-1].unsqueeze(0)
                pre_ub = upper_bounds[-1].unsqueeze(0)
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

                new_layer_lb = (torch.max(kw_lb[layer_idx], out_lbs)).squeeze(0)
                new_layer_ub = (torch.min(kw_ub[layer_idx], out_ubs)).squeeze(0)
            elif type(layer) == nn.ReLU:
                new_layer_lb = F.relu(lower_bounds[-1])
                new_layer_ub = F.relu(upper_bounds[-1])

            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                new_layer_lb = lower_bounds[-1].view(-1)
                new_layer_ub = upper_bounds[-1].view(-1)
            else:
                raise NotImplementedError

            lower_bounds.append(new_layer_lb)
            upper_bounds.append(new_layer_ub)

            layer_idx += 1

        # compare KW_INT bounds with KW bounds.
        # if they are different, reupdate the kw model
        for pre_idx in pre_relu_indices:
            if torch.sum(abs(lower_bounds[pre_idx]-kw_lb[pre_idx])>1e-4)==0 and torch.sum(abs(upper_bounds[pre_idx]-kw_ub[pre_idx])>1e-4)==0:
                pass
            else:
                print(f"initial kw: change_idx at {pre_idx}")
                lower_bounds, upper_bounds, dual_info = self.loose_dual.update_kw_bounds( pre_idx, pre_lb_all = lower_bounds, pre_ub_all = upper_bounds, dual_info = dual_info)
                break
        

        # record the dual_info as an attribute of the loose_dual instance
        # this should be the only dual instance recorded and should not 
        # be modified 
        self.loose_dual.orig_dual = dual_info


        ## NEW STRUCTURE: use the computed bounds to directly introduce gurobi models 

        # Initialize the model 
        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)
        # keep a record of model's information
        self.gurobi_vars = []
        self.relu_constrs = []
        self.relu_indices_mask = []

        ## Do the input layer, which is a special case
        inp_gurobi_vars = []
        zero_var = self.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name= 'zero')
        if input_domain.dim() == 2:
            # This is a linear input.
            for dim, (lb, ub) in enumerate(input_domain):
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.CONTINUOUS,
                                      name=f'inp_{dim}')
                inp_gurobi_vars.append(v)
        else:
            assert input_domain.dim() == 4
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
                        row_vars.append(v)
                    chan_vars.append(row_vars)
                inp_gurobi_vars.append(chan_vars)
        self.model.update()

        self.gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        relu_idx = 0
        layer_idx = 1
        for layer in self.layers:
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                # Get the better estimates from KW and Interval Bounds
                out_lbs = lower_bounds[layer_idx]
                out_ubs = upper_bounds[layer_idx]
                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, self.gurobi_vars[-1])

                    out_lb = out_lbs[neuron_idx].item()
                    out_ub = out_ubs[neuron_idx].item()
                    v = self.model.addVar(lb=out_lb, ub=out_ub, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    new_layer_gurobi_vars.append(v)

            elif type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)
                pre_lb_size = lower_bounds[layer_idx-1].unsqueeze(0).size()
                out_lbs = lower_bounds[layer_idx].unsqueeze(0)
                out_ubs = upper_bounds[layer_idx].unsqueeze(0)
                
                for out_chan_idx in range(out_lbs.size(1)):
                    out_chan_vars = []
                    for out_row_idx in range(out_lbs.size(2)):
                        out_row_vars = []
                        for out_col_idx in range(out_lbs.size(3)):
                            lin_expr = layer.bias[out_chan_idx].item()

                            for in_chan_idx in range(layer.weight.shape[1]):
                                for ker_row_idx in range(layer.weight.shape[2]):
                                    in_row_idx = -layer.padding[0] + layer.stride[0]*out_row_idx + ker_row_idx
                                    if (in_row_idx < 0) or (in_row_idx == pre_lb_size[2]):
                                        # This is padding -> value of 0
                                        continue
                                    for ker_col_idx in range(layer.weight.shape[3]):
                                        in_col_idx = -layer.padding[1] + layer.stride[1]*out_col_idx + ker_col_idx
                                        if (in_col_idx < 0) or (in_col_idx == pre_lb_size[3]):
                                            # This is padding -> value of 0
                                            continue
                                        coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                                        lin_expr += coeff * self.gurobi_vars[-1][in_chan_idx][in_row_idx][in_col_idx]

                            out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            v = self.model.addVar(lb=out_lb, ub=out_ub,
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(v == lin_expr)
                            self.model.update()

                            out_row_vars.append(v)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)

            elif type(layer) == nn.ReLU:
                new_relu_layer_constr = []
                if isinstance(self.gurobi_vars[-1][0], list):
                    # This is convolutional
                    pre_lbs = lower_bounds[layer_idx-1]
                    pre_ubs = upper_bounds[layer_idx-1]
                    new_layer_mask = [] 
                    ratios_all = dual_info[0][layer_idx].d
                    bias_all = -pre_lbs*ratios_all
                    bias_all = bias_all*dual_info[0][layer_idx].I.squeeze(0).float()
                    bias_all = bias_all.squeeze(0)
                    temp = pre_lbs.size()
                    out_chain = temp[0]
                    out_height = temp[1]
                    out_width = temp[2]
                    for chan_idx, channel in enumerate(self.gurobi_vars[-1]):
                        chan_vars = []
                        for row_idx, row in enumerate(channel):
                            row_vars = []
                            for col_idx, pre_var in enumerate(row):
                                slope = ratios_all[0,chan_idx, row_idx, col_idx].item()
                                pre_ub = pre_ubs[chan_idx, row_idx, col_idx].item()
                                bias =  bias_all[chan_idx, row_idx, col_idx].item()

                                if slope==1.0:
                                    # ReLU is always passing
                                    v = pre_var
                                    new_layer_mask.append(1)
                                elif slope==0.0:
                                    v = zero_var
                                    new_layer_mask.append(0)
                                else:
                                    #lb = 0
                                    ub = pre_ub
                                    new_layer_mask.append(-1)
                                    v =self.model.addVar(ub=ub,
                                                          obj=0, vtype=grb.GRB.CONTINUOUS,
                                                          name=f'ReLU{relu_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                    out_idx = col_idx + row_idx*out_width + chan_idx*out_height*out_width
                                    new_relu_layer_constr.append(self.model.addConstr(v >= 0, name= f'ReLU{relu_idx}_{out_idx}_a_0'))
                                    new_relu_layer_constr.append(self.model.addConstr(v >= pre_var, name= f'ReLU{relu_idx}_{out_idx}_a_1'))
                                    new_relu_layer_constr.append(self.model.addConstr(v <= slope*pre_var + bias, f'ReLU{relu_idx}_{out_idx}_a_2'))
                                row_vars.append(v)
                            chan_vars.append(row_vars)
                        new_layer_gurobi_vars.append(chan_vars)
                else:
                    pre_lbs = lower_bounds[layer_idx-1]
                    pre_ubs = upper_bounds[layer_idx-1]
                    new_layer_mask = [] 
                    ratios_all = dual_info[0][layer_idx].d.squeeze(0)
                    bias_all = -pre_lbs*ratios_all
                    bias_all = bias_all*dual_info[0][layer_idx].I.squeeze(0).float()

                    assert isinstance(self.gurobi_vars[-1][0], grb.Var)
                    for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
                        pre_ub = pre_ubs[neuron_idx].item()
                        slope = ratios_all[neuron_idx].item()
                        bias = bias_all[neuron_idx].item()

                        if slope==1.0:
                            # The ReLU is always passing
                            v = pre_var
                            new_layer_mask.append(1)
                        elif slope==0.0:
                            v = zero_var
                            # No need to add an additional constraint that v==0
                            # because this will be covered by the bounds we set on
                            # the value of v.
                            new_layer_mask.append(0)
                        else:
                            ### Trying to introduce lb constraint
                            #lb = 0
                            ub = pre_ub
                            v = self.model.addVar(ub=ub,
                                                  obj=0,
                                                  vtype=grb.GRB.CONTINUOUS,
                                                  name=f'ReLU{layer_idx}_{neuron_idx}')
                            new_relu_layer_constr.append(self.model.addConstr(v >= 0, name=f'ReLU{relu_idx}_{neuron_idx}_a_0'))
                            new_relu_layer_constr.append(self.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx}_{neuron_idx}_a_1'))
                            new_relu_layer_constr.append(self.model.addConstr(v <= slope * pre_var + bias, name=f'ReLU{relu_idx}_{neuron_idx}_a_2'))
                            new_layer_mask.append(-1)

                        new_layer_gurobi_vars.append(v)

                new_relu_mask.append(torch.tensor(new_layer_mask))
                self.relu_constrs.append(new_relu_layer_constr)
                relu_idx += 1

            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                for chan_idx in range(len(self.gurobi_vars[-1])):
                    for row_idx in range(len(self.gurobi_vars[-1][chan_idx])):
                        new_layer_gurobi_vars.extend(self.gurobi_vars[-1][chan_idx][row_idx])
            else:
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

        # Assert that this is as expected a network with a single output
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        self.model.update()
        self.model.setObjective(self.gurobi_vars[-1][0], grb.GRB.MINIMIZE)
        self.model.optimize()
        #assert self.model.status == 2, "LP wasn't optimally solved"
        self.check_optimization_success()
        glb =  self.gurobi_vars[-1][0].X
        lower_bounds[-1] = torch.tensor([glb])
        
        inp_size = lower_bounds[0].size()
        mini_inp = torch.zeros(inp_size)

        if len(inp_size)==1:
            # This is a linear input.
            for i in range(inp_size[0]):
                mini_inp[i] = self.gurobi_vars[0][i].x

        else:
            for i in range(inp_size[0]):
                for j in range(inp_size[1]):
                    for k in range(inp_size[2]):
                        mini_inp[i,j,k] = self.gurobi_vars[0][i][j][k].x
        gub = self.net(mini_inp.unsqueeze(0)).item()

        # record model information
        # indices for undecided relu-nodes
        self.relu_indices_mask = [(i==-1).nonzero().view(-1).tolist() for i in new_relu_mask]
        # flatten high-dimensional gurobi var lists
        for l_idx, layer in enumerate(self.layers):
            if type(layer) is nn.Conv2d:
                flattened_gurobi = []
                for chan_idx in range(len(self.gurobi_vars[l_idx+1])):
                    for row_idx in range(len(self.gurobi_vars[l_idx+1][chan_idx])):
                        flattened_gurobi.extend(self.gurobi_vars[l_idx+1][chan_idx][row_idx])
                self.gurobi_vars[l_idx+1] = flattened_gurobi
                if type(self.layers[l_idx+1]) is nn.ReLU:
                    flattened_gurobi = []
                    for chan_idx in range(len(self.gurobi_vars[l_idx+2])):
                        for row_idx in range(len(self.gurobi_vars[l_idx+2][chan_idx])):
                            flattened_gurobi.extend(self.gurobi_vars[l_idx+2][chan_idx][row_idx])
                    self.gurobi_vars[l_idx+2] = flattened_gurobi
            else:
                continue
        self.replacing_bd_index = len(lower_bounds)

        # get all dual variables
        duals = [torch.zeros(len(i),3) for i in new_relu_mask]
        duals_other = {}

        for i in self.relu_constrs:
            if len(i) == 0:
                break
            else:
                for constr in i:
                    constr_name = constr.ConstrName.split('_')
                    #print(constr_name)
                    if constr_name[-2] == 'a': 
                        layer_idx = int(constr_name[0][4:])
                        cons_idx = int(constr_name[-1]) 
                        node_idx = int(constr_name[1])
                        duals[layer_idx][node_idx][cons_idx] = constr.getAttr('Pi')
                        #print(layer_idx, node_idx, cons_idx)
                        #print(constr.getAttr('Pi'))
                    else:
                        duals_other[constr.ConstrName] = constr.getAttr('Pi')

        primals = []
        for one_layer in self.gurobi_vars[1:]:
            temp = []
            for var in one_layer:
                temp.append(var.x)
            primals.append(temp)
        return gub, glb, mini_inp.unsqueeze(0), duals, duals_other, primals, new_relu_mask, lower_bounds, upper_bounds, pre_relu_indices


    def update_the_model(self, relu_mask, pre_lb_all, pre_ub_all, decision, choice):
        '''
        The model updates upper and lower bounds after introducing a relu constraint and then update the gurobi model
        using these updated bounds

        input:
        relu_mask: the copied mask of the parent domain, 
        pre_lb, pre_ub: lower and upper bounds of the parent domain
        decision: the index of the node where we make branching decision
        choice: force no-passing constraint (0) or all passing constraint (1)
        pre_relu_indices: indices of bounds that the layers prior to a relu_layer
        
        output: global lower bound, updated mask, updated lower and upper bounds
        '''
        
        # modifying the mask according to the branching decision and choice made
        relu_mask[decision[0]][decision[1]] = choice
        
        # Computing updated KW bounds
        # first changed_bounds_index should be the index of 
        # the layer right before the relu layer we decide to split on
        first_changed_bounds_index = self.loose_dual.pre_relu_indices[decision[0]]
        self.replacing_bd_index = min(self.replacing_bd_index, first_changed_bounds_index)
        lower_bounds, upper_bounds, _= self.loose_dual.update_kw_bounds(self.replacing_bd_index, pre_lb_all, pre_ub_all, decision, choice)
        ## DEBUG
        #lower_init, upper_init, _ = self.loose_dual.init_kw_bounds(pre_lb_all, pre_ub_all, decision, choice)
        #for i in range(len(lower_init)):
        #    assert torch.sum(torch.abs(lower_init[i]-lower_bounds[i])) < 1e-3, 'lower is wrong'
        #    assert torch.sum(torch.abs(upper_init[i]-upper_bounds[i])) < 1e-3, 'upper is wrong'
        # compute interval bounds
        change_idx = len(lower_bounds)
        #assert change_idx==10, 'wrong'

        inter_bounds_index = first_changed_bounds_index+2 

        
        for layer in self.layers[first_changed_bounds_index+1:]:
            if type(layer) is nn.Linear:
                pre_lb = lower_bounds[inter_bounds_index-1]
                pre_ub = upper_bounds[inter_bounds_index-1]
                pos_w = torch.clamp(layer.weight, 0, None)
                neg_w = torch.clamp(layer.weight, None, 0)
                out_lbs = pos_w @ pre_lb + neg_w @ pre_ub + layer.bias
                out_ubs = pos_w @ pre_ub + neg_w @ pre_lb + layer.bias
                # Get the better estimates from KW and Interval Bounds
                if torch.sum(lower_bounds[inter_bounds_index]>=out_lbs).item()!= len(lower_bounds[inter_bounds_index]):
                    lower_bounds[inter_bounds_index] = torch.max(lower_bounds[inter_bounds_index], out_lbs)
                    change_idx = min(inter_bounds_index, change_idx)

                if torch.sum(upper_bounds[inter_bounds_index]<=out_ubs).item()!= len(upper_bounds[inter_bounds_index]):
                    upper_bounds[inter_bounds_index] = torch.min(upper_bounds[inter_bounds_index], out_ubs)
                    change_idx = min(inter_bounds_index, change_idx)

            elif type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)
                pre_lb = lower_bounds[inter_bounds_index-1].unsqueeze(0)
                pre_ub = upper_bounds[inter_bounds_index-1].unsqueeze(0)
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

                #lower_bounds[inter_bounds_index] = torch.max(lower_bounds[inter_bounds_index], out_lbs).squeeze(0)
                #upper_bounds[inter_bounds_index] = torch.min(upper_bounds[inter_bounds_index], out_ubs).squeeze(0)

                if torch.sum(lower_bounds[inter_bounds_index]>=out_lbs).item()!= len(lower_bounds[inter_bounds_index].view(-1)):

                    lower_bounds[inter_bounds_index] = torch.max(lower_bounds[inter_bounds_index], out_lbs).squeeze(0)
                    change_idx = min(inter_bounds_index, change_idx)

                if torch.sum(upper_bounds[inter_bounds_index]<=out_ubs).item()!= len(upper_bounds[inter_bounds_index].view(-1)):

                    upper_bounds[inter_bounds_index] = torch.min(upper_bounds[inter_bounds_index], out_ubs).squeeze(0)
                    change_idx = min(inter_bounds_index, change_idx)

            elif type(layer) == nn.ReLU:
                lower_bounds[inter_bounds_index] = F.relu(lower_bounds[inter_bounds_index-1])
                upper_bounds[inter_bounds_index] = F.relu(upper_bounds[inter_bounds_index-1])

            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                lower_bounds[inter_bounds_index] = lower_bounds[inter_bounds_index-1].view(-1)
                upper_bounds[inter_bounds_index] = upper_bounds[inter_bounds_index-1].view(-1)
            else:
                raise NotImplementedError
            inter_bounds_index += 1

        if change_idx < len(lower_bounds)-1:
            print(f'update_kw interval is better: change_idx at {change_idx}')
            #lower_init, upper_init, _ = self.loose_dual.init_kw_bounds(lower_bounds, upper_bounds)
            lower_bounds, upper_bounds, _= self.loose_dual.update_kw_bounds(self.replacing_bd_index, lower_bounds, upper_bounds)
            ## DEBUG
            #for i in range(len(lower_init)-1):
            #    assert torch.sum(torch.abs(lower_init[i]-lower_bounds_k[i])) < 1e-3, 'change lower is wrong'
            #    assert torch.sum(torch.abs(upper_init[i]-upper_bounds_k[i])) < 1e-3, 'change upper is wrong'

            #lower_bounds = lower_bounds_k
            #upper_bounds = upper_bounds_k

        # reintroduce ub and lb for gurobi constraints
        introduced_constrs_all = [] 
        rep_index = self.replacing_bd_index
        for layer in self.layers[self.replacing_bd_index-1:]:
            if type(layer) is nn.Linear:
                for idx, var in enumerate(self.gurobi_vars[rep_index]):
                    var.ub = upper_bounds[rep_index][idx].item()
                    var.lb = lower_bounds[rep_index][idx].item()
                #self.model_lower_bounds[rep_index] = lower_bounds[rep_index].clone()
                #self.model_upper_bounds[rep_index] = upper_bounds[rep_index].clone()

            elif type(layer) is nn.Conv2d:
                conv_ub = upper_bounds[rep_index].view(-1)
                conv_lb = lower_bounds[rep_index].view(-1)
                for idx, var in enumerate(self.gurobi_vars[rep_index]):
                    var.ub = conv_ub[idx].item()
                    var.lb = conv_lb[idx].item()
                #self.model_lower_bounds[rep_index] = lower_bounds[rep_index].clone()
                #self.model_upper_bounds[rep_index] = upper_bounds[rep_index].clone()

            elif type(layer) is nn.ReLU:
                layer_introduced_ctrs = []
                # locate relu index and remove all associated constraints
                relu_idx = self.loose_dual.pre_relu_indices.index(rep_index-1)
                #remove relu constraints
                self.model.remove(self.relu_constrs[relu_idx])
                self.relu_constrs[relu_idx] = []
                # reintroduce relu constraints
                pre_lbs = lower_bounds[rep_index-1].view(-1)
                pre_ubs = upper_bounds[rep_index-1].view(-1)
                for unmasked_idx in self.relu_indices_mask[relu_idx]:
                    pre_lb = pre_lbs[unmasked_idx].item()
                    pre_ub = pre_ubs[unmasked_idx].item()
                    var = self.gurobi_vars[rep_index][unmasked_idx]
                    pre_var = self.gurobi_vars[rep_index-1][unmasked_idx]

                    if pre_lb >= 0 and pre_ub >= 0:
                        # ReLU is always passing
                        var.lb = pre_lb
                        var.ub = pre_ub
                        layer_introduced_ctrs.append(self.model.addConstr(pre_var == var, name=f'ReLU{relu_idx}_{unmasked_idx}_p'))
                        relu_mask[relu_idx][unmasked_idx] = 1
                    elif pre_lb <= 0 and pre_ub <= 0:
                        #var.lb = 0
                        #var.ub = 0 
                        layer_introduced_ctrs.append(self.model.addConstr( var==0, name=f'ReLU{relu_idx}_{unmasked_idx}_p'))
                        relu_mask[relu_idx][unmasked_idx] = 0
                    else:
                        #assert relu_mask[relu_idx][unmasked_idx]==-1,'mask is wrong'
                        #var.lb = 0
                        var.ub = pre_ub
                        layer_introduced_ctrs.append(self.model.addConstr(var >= 0, name=f'ReLU{relu_idx}_{unmasked_idx}_a_0'))
                        layer_introduced_ctrs.append(self.model.addConstr(var >= pre_var, name=f'ReLU{relu_idx}_{unmasked_idx}_a_1'))
                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        layer_introduced_ctrs.append(self.model.addConstr(var <= slope*pre_var + bias, name=f'ReLU{relu_idx}_{unmasked_idx}_a_2'))
                introduced_constrs_all.append(layer_introduced_ctrs)

            elif type(layer) is View:
                pass
            elif type(layer) is Flatten:
                pass
            else: 
                raise NotImplementedError
            self.model.update()
            rep_index += 1

         # compute optimum
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        self.model.update()
        #self.model.reset()
        self.model.setObjective(self.gurobi_vars[-1][0], grb.GRB.MINIMIZE)
        self.model.optimize()
        #assert self.model.status == 2, "LP wasn't optimally solved"
        self.check_optimization_success(introduced_constrs_all)
        glb = self.gurobi_vars[-1][0].X
        lower_bounds[-1] = torch.tensor([glb])
        
        # get input variable values at which minimum is achieved
        inp_size = lower_bounds[0].size()
        mini_inp = torch.zeros(inp_size)
        if len(inp_size)==1:
            # This is a linear input.
            for i in range(inp_size[0]):
                mini_inp[i] = self.gurobi_vars[0][i].x

        else:
            for i in range(inp_size[0]):
                for j in range(inp_size[1]):
                    for k in range(inp_size[2]):
                        mini_inp[i,j,k] = self.gurobi_vars[0][i][j][k].x
        gub = self.net(mini_inp.unsqueeze(0)).item()

        # get all dual variables
        duals = [torch.zeros(len(i),3) for i in relu_mask]
        duals_other = {}

        for case in [self.relu_constrs, introduced_constrs_all]:
            for i in case:
                if len(i) == 0:
                    break
                else:
                    for constr in i:
                        constr_name = constr.ConstrName.split('_')
                        #print(constr_name)
                        if constr_name[-2] == 'a': 
                            layer_idx = int(constr_name[0][4:])
                            cons_idx = int(constr_name[-1]) 
                            node_idx = int(constr_name[1])
                            duals[layer_idx][node_idx][cons_idx] = constr.getAttr('Pi')
                            #print(layer_idx, node_idx, cons_idx)
                            #print(constr.getAttr('Pi'))
                        else:
                            duals_other[constr.ConstrName] = constr.getAttr('Pi')

        primals = []
        for one_layer in self.gurobi_vars[1:]:
            temp = []
            for var in one_layer:
                temp.append(var.x)
            primals.append(temp)


        # remove introduced vars and constraints
        for introduced_cons_layer in introduced_constrs_all:
            self.model.remove(introduced_cons_layer)

        self.model.update()

        return gub, glb, mini_inp.unsqueeze(0), duals, duals_other, primals, relu_mask, lower_bounds, upper_bounds 




