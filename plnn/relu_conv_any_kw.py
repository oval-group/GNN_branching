import torch
import copy

from plnn.branch_and_bound import pick_out, add_domain, prune_domains
from torch import nn
from plnn.kw_score_conv import choose_node_conv
import time
#from graphnet.graph_score import GraphChoice


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
    def __init__(self, mask,  lb=-float('inf'), ub=float('inf'), lb_all=None, up_all = None):
        self.mask = mask
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all
        self.upper_all = up_all
     


    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound


def relu_bab(net, domain,  x, ball_eps, eps=1e-4, pgd_threshold = 1, split_decision='kw', sparsest_layer=0, decision_bound=None, linear=False, model_path=None, bounded = True):
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
    nb_visited_states = 0
    #global_ub_point, global_ub = net.get_upper_bound(domain)

    global_ub, global_lb, global_ub_point, updated_mask, lower_bounds, upper_bounds, pre_relu_indices = net.build_the_model(domain, x, ball_eps, bounded)

    print(global_lb)
    #import pdb; pdb.set_trace()
    if global_lb > 0:
        return global_lb, global_ub, global_ub_point, nb_visited_states

    candidate_domain = ReLUDomain(updated_mask, global_lb, global_ub, lower_bounds, upper_bounds)
    domains = [candidate_domain]
    tot_ambi_nodes = 0
    for layer_mask in updated_mask: 
        tot_ambi_nodes += torch.sum(layer_mask ==-1).item()
    
    ambi_nodes_threshold = (1-pgd_threshold) * tot_ambi_nodes


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
    if split_decision != 'kw':
        graph = GraphChoice(updated_mask, model_path, linear=linear)
    else:
        icp_score = 0

     

    while global_ub - global_lb > eps:
        # Pick a domain to branch over and remove that from our current list of
        # domains. Also, potentially perform some pruning on the way.
        candidate_domain = pick_out(domains, global_ub - eps)
        # Generate new, smaller domains by splitting over a ReLU
        mask = candidate_domain.mask
        orig_lbs = candidate_domain.lower_all
        orig_ubs = candidate_domain.upper_all
        if split_decision == 'kw':
            branching_decision, icp_score = choose_node_conv(orig_lbs, orig_ubs, mask, net.layers, pre_relu_indices, icp_score, random_order, sparsest_layer)
        else:
            branching_decision = graph.decision(orig_lbs, orig_ubs, mask, net.layers)


        print(f'splitting decision: {branching_decision}')
        relu_start = time.time()
        for choice in [0,1]:
            # Find the upper and lower bounds on the minimum in the domain
            # defined by n_mask_i
            nb_visited_states += 1
            if (nb_visited_states % 10) == 0:
                print(f"Running Nb states visited: {nb_visited_states}")
            
            mask_temp = [i.clone() for i in mask]
            dom_ub,dom_lb, dom_ub_point, updated_mask, dom_lb_all, dom_ub_all = net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, branching_decision, choice)
            #mask_temp = [i.clone() for i in mask]
            #dom_ub_1,dom_lb_1, dom_ub_point_1, updated_mask_1, dom_lb_all_1, dom_ub_all_1, dom_dual_info= net.get_lower_bound( mask_temp, orig_lbs, orig_ubs, branching_decision, choice, None)
            #assert dom_ub_all[-1][0] ==dom_ub_all_1[-1][0], "up is wrong"
            #assert dom_lb ==dom_lb_1, "lower is wrong"

            if dom_ub < global_ub:
                global_ub = dom_ub
                global_ub_point = dom_ub_point
            
            # if global_ub is greater than 0 and a certain percentage 
            # (pgd_threshold) of relus are fixed, we start to use pgd 
            # to find better upper bounds. That is once the number of 
            # current ambiguous is smaller than ambi_nodes_threshold 
            
            current_tot_ambi_nodes = 0
            for layer_mask in updated_mask:
                current_tot_ambi_nodes += torch.sum(layer_mask == -1).item()

            #import pdb; pdb.set_trace()            
            if global_ub >0  and current_tot_ambi_nodes < ambi_nodes_threshold:
                _, global_ub = net.get_upper_bound_pgd(orig_lbs[0], orig_ubs[0], dom_ub_point)

            print('dom_lb: ', dom_lb)
            print('dom_ub: ', dom_ub)

            
            if dom_lb < global_ub:
                dom_to_add = ReLUDomain(updated_mask, lb=dom_lb, ub= dom_ub, lb_all= dom_lb_all, up_all = dom_ub_all)
                add_domain(dom_to_add, domains)
                prune_counter += 1

        relu_end = time.time()
        print('one relu split requires (KW): ', relu_end - relu_start)

        if prune_counter >= 100 and len(domains) >= 100:
            domains = prune_domains(domains, global_ub - eps)
            prune_counter = 0

        if len(domains) > 0:
            global_lb = domains[0].lower_bound
        else:
            # If there is no more domains, we have pruned them all
            global_lb = global_ub - eps

        print(f"Current: lb:{global_lb}\t ub: {global_ub}")

        # Stopping criterion
        if (decision_bound is not None) and (global_lb >= decision_bound):
            break
        elif global_ub < decision_bound:
            break

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
