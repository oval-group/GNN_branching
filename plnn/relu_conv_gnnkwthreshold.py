import torch
import copy

from plnn.branch_and_bound import pick_out, add_domain, prune_domains
from torch import nn
from plnn.kw_score_conv import choose_node_conv
from graphnet.graph_score import GraphChoice
import time

'''
this file mainly use gnn to make branching decision. 
When the improvemen done by gnn is less than a provided threshold, we switch to kw

NOTE: changed bab to assume that decision bound is definitely provided
'''

path = '/home/jodie/PLNN/PLNN-verification-private/cifar_exp_all/gnnkwthreshold/'

class ReLUDomain:
    '''
    Object representing a domain where the domain is specified by decision
    assigned to ReLUs.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.

    The domain is specified by `mask` which corresponds to a pattern of ReLUs.
    Neurons mapping to a  0 value are assumed to always have negative input (0 output slope)
          "               1                    "             positive input (1 output slope).
          "               -1 value are considered free and have no assumptions.

    For a MaxPooling unit, -1 indicates that we haven't picked a dominating input
    Otherwise, this indicates which one is the dominant one
    '''
    def __init__(self, mask,  lb=-float('inf'), ub=float('inf'), lb_all=None, up_all = None, gnn_decision=None):
        self.mask = mask
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all
        self.upper_all = up_all
        self.gnn_decision = gnn_decision
        #self.kw_decision = kw_decision
     


    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound


def relu_gnn(net, domain,  x, ball_eps, bounded, eps=1e-4, pgd_threshold = 1, sparsest_layer=0, branching_threshold = 0.2, kwbd_threshold=10, split_decision=None, decision_bound=None, linear=False, model_path=None, dump_trace = None):
    '''
    Uses branch and bound algorithm to evaluate the global minimum
    of a given neural network.
    `net`           : Neural Network class, defining the `get_upper_bound` and
                      `get_lower_bound` functions, supporting the `mask` argument
                      indicating the phase of the ReLU.
    `eps`           : Maximum difference between the UB and LB over the minimum
                      before we consider having converged
    `decision_bound`: If not None, stop the search if the UB and LB are both
                      superior or both inferior to this value.
    `pgd_threshold` : Once the number of relus being fixed during the algorithm
                      is above pdg_threshold percentage of the total ambiguous nodes
                      at the beginning, we initiate pgd attacks to find 
                      a better global upper bound

    Returns         : Lower bound and Upper bound on the global minimum,
                      as well as the point where the upper bound is achieved
    '''
    
    if dump_trace is not None:
        f = open(path + dump_trace, 'w')

    nb_visited_states = 0
    #global_ub_point, global_ub = net.get_upper_bound(domain)
    global_ub, global_lb, global_ub_point, dual_vars, dual_vars_other, primals, updated_mask, lower_bounds_all, upper_bounds_all, pre_relu_indices= net.build_the_model(domain, x, ball_eps, bounded)

    if global_lb>=decision_bound or global_ub<decision_bound:
        return global_lb, global_ub, global_ub_point, nb_visited_states

    
    if dump_trace is not None:
        f.write(f'{global_lb}\n')
    print(global_lb)
    

    # assign layer priority for each layer. This ranking is 
    # used for making random choice with preferences when kw-score fails
    # here we have an increased preference for later layers while
    # giving the least preference to the sparsest layer
    random_order = list(range(len(updated_mask)))
    try: 
        random_order.remove(sparsest_layer)
        random_order = [sparsest_layer]+random_order
    except:
        pass

    #import pdb; pdb.set_trace()
    prune_counter = 0
    # initialize graphnet

    graph = GraphChoice(updated_mask, model_path, linear=linear)
    bounds_indices = [0]+pre_relu_indices + [len(net.layers)]
    layers = {}; 
    layers['fixed_layers'] =[copy.deepcopy(i).cuda() for i in net.layers[:-1]] 
    layers['prop_layers'] = [copy.deepcopy(net.layers[-1]).cuda()]

    lower_bounds_graph = [lower_bounds_all[i].unsqueeze(0) for i in bounds_indices]
    upper_bounds_graph = [upper_bounds_all[i].unsqueeze(0) for i in bounds_indices]
    gnn_decision = graph.decision(lower_bounds_graph, upper_bounds_graph, dual_vars,global_ub_point, primals, layers, updated_mask)

    icp_score = 0

    candidate_domain = ReLUDomain(updated_mask, global_lb, global_ub, lower_bounds_all, upper_bounds_all, gnn_decision=gnn_decision)
    domains = [candidate_domain]
    
    ineff_kw_dc = {}

    while global_ub - global_lb > eps:
        # Pick a domain to branch over and remove that from our current list of
        # domains. Also, potentially perform some pruning on the way.
        candidate_domain = pick_out(domains, global_ub - eps)
        # Generate new, smaller domains by splitting over a ReLU
        branching_decision = candidate_domain.gnn_decision
        lower_bound = candidate_domain.lower_bound
        mask = candidate_domain.mask
        orig_lbs = candidate_domain.lower_all
        orig_ubs = candidate_domain.upper_all


        # Find the upper and lower bounds on the minimum in the domain
        nb_visited_states += 2
        if (nb_visited_states % 10) == 0:
            print(f"Running Nb states visited: {nb_visited_states}")

         
        # first compute gnn decision
        mask_temp = [i.clone() for i in mask]
        dom_ub, dom_lb, dom_ub_point, dom_dual_vars, dom_dual_vars_other, dom_primals, dom_updated_mask, dom_lb_all, dom_ub_all = net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, branching_decision, 0)
        mask_temp = [i.clone() for i in mask]
        dom_ub1, dom_lb1, dom_ub_point1, dom_dual_vars1, dom_dual_vars_other1, dom_primals1, dom_updated_mask1, dom_lb_all1, dom_ub_all1 = net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, branching_decision, 1)
        
        # compare with branching_threshold
        gnn_improvement = (min(dom_lb,0) + min(dom_lb1, 0) -2*lower_bound)/(-2*lower_bound)

        kw_improvement = -1 ; kw_decision = None
        
        if gnn_improvement < branching_threshold:
            # use kw decision
            kw_decision, icp_score = choose_node_conv(orig_lbs, orig_ubs, mask, net.layers, pre_relu_indices, icp_score, random_order, sparsest_layer)

            # if the kw decision is labeled as a bad point, skip this kw decison
            try:
                ineff_kw_counts = ineff_kw_dc[f'{kw_decision[0]}-{kw_decision[1]}']
            except KeyError:
                ineff_kw_counts = 0

            print(f'gnn below branching_threshold, trying kw decision {kw_decision} with ineff-counts {ineff_kw_counts}')
                
            if ineff_kw_counts < kwbd_threshold:
                mask_temp = [i.clone() for i in mask]
                dom_ub_kw,dom_lb_kw, dom_ub_point_kw, dom_dual_vars_kw, dom_dual_vars_other_kw, dom_primals_kw, dom_updated_mask_kw, dom_lb_all_kw, dom_ub_all_kw = net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, kw_decision, 0)
                mask_temp = [i.clone() for i in mask]
                dom_ub1_kw,dom_lb1_kw, dom_ub_point1_kw, dom_dual_vars1_kw, dom_dual_vars_other1_kw, dom_primals1_kw, dom_updated_mask1_kw, dom_lb_all1_kw, dom_ub_all1_kw = net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, kw_decision, 1)
                
                kw_improvement = (min(dom_lb_kw,0) + min(dom_lb1_kw, 0) -2*lower_bound)/(-2*lower_bound)
                
                if kw_improvement < gnn_improvement and kw_improvement < 0.05:
                    ## record bad kw points
                    try:
                        ineff_kw_dc[f'{kw_decision[0]}-{kw_decision[1]}'] +=1
                    except KeyError:
                        ineff_kw_dc[f'{kw_decision[0]}-{kw_decision[1]}'] = 1
                elif kw_improvement > gnn_improvement:
                    ## using kw decision instead
                    branching_decision = kw_decision
                    print('using kw decision')
                    dom_ub = dom_ub_kw; dom_lb = dom_lb_kw; dom_ub_point = dom_ub_point_kw; 
                    dom_dual_vars = dom_dual_vars_kw; dom_dual_vars_other = dom_dual_vars_other_kw;
                    dom_primals = dom_primals_kw; dom_updated_mask = dom_updated_mask_kw;
                    dom_lb_all = dom_lb_all_kw; dom_ub_all = dom_ub_all_kw

                    dom_ub1 = dom_ub1_kw; dom_lb1 = dom_lb1_kw; dom_ub_point1 = dom_ub_point1_kw; 
                    dom_dual_vars1 = dom_dual_vars1_kw; dom_dual_vars_other1 = dom_dual_vars_other1_kw;
                    dom_primals1 = dom_primals1_kw; dom_updated_mask1 = dom_updated_mask1_kw;
                    dom_lb_all1 = dom_lb_all1_kw; dom_ub_all1 = dom_ub_all1_kw
                else:
                    pass
            else:
                print(f'encountered a suspecious KW point, using gnn instead')



        if dump_trace is not None:
            f.write(f'branch {nb_visited_states} decision {branching_decision} gnn: improvement {gnn_improvement} decision {candidate_domain.gnn_decision} kw: improvement {kw_improvement} decision {kw_decision}\n')
        
        print(f'branch {nb_visited_states} decision {branching_decision} gnn: improvement {gnn_improvement} decision {candidate_domain.gnn_decision} kw: improvement {kw_improvement} decision {kw_decision}')

        # update global upper bound


        if dom_ub < global_ub:
            global_ub = dom_ub
            global_ub_point = dom_ub_point
        if dom_ub1 < global_ub:
            global_ub = dom_ub1
            global_ub_point = dom_ub_point1

            
            # if global_ub is greater than 0 and a certain percentage 
            # (pgd_threshold) of relus are fixed, we start to use pgd 
            # to find better upper bounds. That is once the number of 
            # current ambiguous is smaller than ambi_nodes_threshold 
            

        print('dom_lb: ', dom_lb, dom_lb1)
        print('dom_ub: ', dom_ub, dom_ub1)

            
        if dom_lb < decision_bound:
            dom_lb_graph = [dom_lb_all[i].unsqueeze(0) for i in bounds_indices]
            dom_ub_graph = [dom_ub_all[i].unsqueeze(0) for i in bounds_indices]
            dom_gnn_decision = graph.decision(dom_lb_graph, dom_ub_graph, dom_dual_vars, dom_ub_point, dom_primals, layers, dom_updated_mask)

            dom_to_add = ReLUDomain(dom_updated_mask, lb=dom_lb, ub= dom_ub, lb_all= dom_lb_all, up_all = dom_ub_all,  gnn_decision = dom_gnn_decision)
            add_domain(dom_to_add, domains)
            prune_counter += 1

        if dom_lb1 < decision_bound:
            dom_lb_graph1 = [dom_lb_all1[i].unsqueeze(0) for i in bounds_indices]
            dom_ub_graph1 = [dom_ub_all1[i].unsqueeze(0) for i in bounds_indices]
            dom_gnn_decision1 = graph.decision(dom_lb_graph1, dom_ub_graph1, dom_dual_vars1, dom_ub_point1, dom_primals1, layers, dom_updated_mask1)

            dom_to_add = ReLUDomain(dom_updated_mask1, lb=dom_lb1, ub= dom_ub1, lb_all= dom_lb_all1, up_all = dom_ub_all1, gnn_decision = dom_gnn_decision1)
            add_domain(dom_to_add, domains)
            prune_counter += 1

        if prune_counter >= 100 and len(domains) >= 100:
            domains = prune_domains(domains, global_ub - eps)
            prune_counter = 0

        if len(domains) > 0:
            global_lb = domains[0].lower_bound
        else:
            # If there is no more domains, we have pruned them all
            global_lb = global_ub - eps

        print(f"Current: lb:{global_lb}\t ub: {global_ub}\n")
        if dump_trace is not None:
            f.write(f'{global_lb}\n')

        # Stopping criterion
        if (decision_bound is not None) and (global_lb >= decision_bound):
            break
        elif global_ub < decision_bound:
            break

    if dump_trace is not None:
        f.close()

    return global_lb, global_ub, global_ub_point, nb_visited_states



def kw_split(net, candidate_domain):
    mask = candidate_domain.mask
    orig_lbs = candidate_domain.lower_all_pa
    orig_ubs = candidate_domain.upper_all_pa
    decision = choose_dim(orig_lbs, orig_ubs, mask, net.layers)
    mask_temp_1 = [i.copy() for i in mask]
    mask_temp_1[decision[0]][decision[1]]= 0
    mask_temp_2 = [i.copy() for i in mask]
    mask_temp_2[decision[0]][decision[1]]= 1
    print(f'idx: {decision}')
    all_new_masks = [mask_temp_1, mask_temp_2]
    return all_new_masks




def relu_split(layers, mask):
    '''
    Given a mask that defines a domain, split it according to a non-linerarity.

    The non-linearity is chosen to be as early as possible in the network, but
    this is just a heuristic.

    `layers`: list of layers in the network. Allows us to distinguish
              Maxpooling and ReLUs
    `mask`: A list of [list of {-1, 0, 1}] where each elements corresponds to a layer,
            giving constraints on the Neuron.
    Returns: A list of masks, in the same format

    '''
    done_split = False
    non_lin_layer_idx = 0
    all_new_masks = []
    for layer_idx, layer in enumerate(layers):
        if type(layer) in [nn.ReLU, nn.MaxPool1d]:
            non_lin_lay_mask = mask[non_lin_layer_idx]
            if done_split:
                # We have done our split, so no need for any additional split
                # -> Pass along all of the stuff
                for new_mask in all_new_masks:
                    new_mask.append(non_lin_lay_mask)
            elif all([neuron_dec != -1 for neuron_dec in non_lin_lay_mask]):
                # All the neuron in this layer have already an assumption.
                # This will just be passed along when we do our split.
                pass
            else:
                # This is the first layer we encounter that is not completely
                # assumed so we will take the first "undecided" neuron and
                # split on it.

                # Start by making two copies of everything that came before
                if type(layer) is nn.ReLU:
                    all_new_masks.append([])
                    all_new_masks.append([])
                elif type(layer) is nn.MaxPool1d:
                    for _ in range(layer.kernel_size):
                        all_new_masks.append([])
                else:
                    raise NotImplementedError

                for prev_lay_mask in mask[:non_lin_layer_idx]:
                    for new_mask in all_new_masks:
                        new_mask.append(prev_lay_mask)

                # Now, deal with the layer that we are actually splitting
                neuron_to_flip = non_lin_lay_mask.index(-1)
                for choice, new_mask in enumerate(all_new_masks):
                    # choice will be 0,1 for ReLU
                    # it will be 0, .. kernel_size-1 for MaxPool1d
                    mod_layer = non_lin_lay_mask[:]
                    mod_layer[neuron_to_flip] = choice
                    new_mask.append(mod_layer)

                done_split = True
            non_lin_layer_idx += 1
    for new_mask in all_new_masks:
        assert len(new_mask) == len(mask)
    if not done_split:
        all_new_masks = [mask]
    return all_new_masks
