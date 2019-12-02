#.!/usr/bin/env python
import argparse
#from plnn.relu_bnb_com import com_bab
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn 
import torch
from plnn.conv_kwinter_gen import KWConvGen
from plnn.relu_conv_any_kw import relu_bab
from plnn.relu_conv_gnnkwthreshold import relu_gnn
from plnn.relu_conv_online import relu_online
from exp_utils.model_utils import load_cifar_1to1_exp
from plnn.conv_kwinter_kw import KWConvNetwork
from plnn.mip_solver import MIPNetwork
import time
import pandas as pd
import multiprocessing
import os

'''
This script supports following verifications methods.

1. MIPplanet backed by the commercial solver Gurobi (--gurobi)
2. Branch and Bound with a heuristic splitting strategy, developed based on Kolter and Wong's paper (--bab_kw)
3. Branch and Bound with a GNN splitting strategy (--bab_gnn)
4. Branch and Bound with a online-GNN splitting strategy (--bab_online)

'''

# Pre-fixed parameters
pref_branching_thd = 0.2
pref_online_thd = 2
pref_kwbd_thd = 20

models = {}
models['cifar_base_hinge10_norm_wd1e-4'] = './models/cifar_trained_gnn/best_snapshot_None_0_val_acc_0.826_loss_val_0.1036_epoch_57.pt'


def bab(verif_layers, domain, x, eps_temp, branching,linear, model_name, bounded, return_dict):
    epsilon=1e-4 
    decision_bound=0
    pgd_threshold = 1.
    network = KWConvNetwork(verif_layers)
    try:
        min_lb, min_ub, ub_point, nb_states = relu_bab(network, domain, x, eps_temp, epsilon, pgd_threshold=pgd_threshold, split_decision =branching, decision_bound=decision_bound, linear=linear, model_path = model_name, bounded = bounded)
    except:
        min_lb = None; min_ub = None; ub_point = None; nb_states =None
    return_dict["min_lb"] = min_lb  
    return_dict["min_ub"] = min_ub  
    return_dict["ub_point"] = ub_point  
    return_dict["nb_states"] = nb_states  

def bab_online(verif_layers, domain, x, eps_temp, branching_thd, online_thd, bounded, branching,linear, model_name, trace_name, return_dict):
    epsilon=1e-4 
    decision_bound=0
    pgd_threshold = 1
    network = KWConvGen(verif_layers)
    try:
        min_lb, min_ub, ub_point, nb_states = relu_online(network, domain, x, eps_temp, bounded, epsilon, pgd_threshold=pgd_threshold,branching_threshold=branching_thd, online_threshold=online_thd, split_decision =branching, decision_bound=decision_bound, linear=linear, model_path = model_name, dump_trace = trace_name)
    except:
        min_lb = None; min_ub = None; ub_point = None; nb_states = None
    return_dict["min_lb"] = min_lb  
    return_dict["min_ub"] = min_ub  
    return_dict["ub_point"] = ub_point  
    return_dict["nb_states"] = nb_states  


def bab_gnn(verif_layers, domain, x, eps_temp, branching_thd, bounded, branching,linear, model_name, trace_name, return_dict):
    epsilon=1e-4 
    decision_bound=0
    pgd_threshold = 1
    network = KWConvGen(verif_layers)
    try:
        min_lb, min_ub, ub_point, nb_states = relu_gnn(network, domain, x, eps_temp, bounded, epsilon, pgd_threshold=pgd_threshold,branching_threshold=branching_thd, split_decision =branching, decision_bound=decision_bound, linear=linear, model_path = model_name, dump_trace = trace_name)
    except:
        min_lb = None; min_ub = None; ub_point = None; nb_states = None
    return_dict["min_lb"] = min_lb  
    return_dict["min_ub"] = min_ub  
    return_dict["ub_point"] = ub_point  
    return_dict["nb_states"] = nb_states  
    
def gurobi(verif_layers, domain,  return_dict):
    mip_network = MIPNetwork(verif_layers)
    #mip_binary_network.setup_model(inp_domain, x=x.view(1, -1), ball_eps = eps_temp, bounds=bounds)
    mip_network.setup_model(domain, use_obj_function=True, bounds="interval-kw")
    #mip_network.setup_model(domain, use_obj_function=False, bounds="interval-kw")
    sat, solution, nb_states = mip_network.solve(domain)
    return_dict["out"] = sat  
    return_dict["nb_states"] = nb_states  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='file to save results')
    parser.add_argument('--record_name', type =str, help='file to save results')
    parser.add_argument('--pdprops', type =str, help='pandas table with all props we are interested in')
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--cpus_total', type = int, help='total number of cpus used')
    parser.add_argument('--cpu_id', type=int, help='the index of the cpu from 0 to cpus_total')
    #parser.add_argument('--branching',  type=str, choices = ['kw', 'graph'] )
    parser.add_argument('--model_name', type=str,  default = 'cifar_base_hinge10_norm_wd1e-4', help='GNN model name')
    parser.add_argument('--nn_name', type=str, help='network architecture name')
    parser.add_argument('--data', type=str, default='cifar')
    parser.add_argument('--bab_kw', action='store_true')
    parser.add_argument('--bab_gnn', action='store_true')
    parser.add_argument('--bab_online', action='store_true')
    parser.add_argument('--gurobi', action='store_true')
    parser.add_argument('--testing', action='store_true')
    args = parser.parse_args()

    # initialize a file to record all results, record should be a pandas dataframe
    if args.data == 'cifar':
        path = './cifar_exp/'
        result_path = './cifar_results/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        raise NotImplementedError

    # load all properties
    gt_results = pd.read_pickle(path + args.pdprops)
    bnb_ids = gt_results.index
    assert args.cpu_id < args.cpus_total, 'cpu index exceeds total cpus available'
    batch_size = len(bnb_ids)//args.cpus_total +1
    start_id = args.cpu_id*batch_size
    end_id = min(len(bnb_ids), start_id+batch_size)
    batch_ids = bnb_ids[start_id: end_id]  

    nn_type = args.nn_name.split('_')[-1] 
    

    if args.record:
        if args.record_name is not None:
            record_name = args.record_name
        else:
            method_name = ''
            columns = ["Idx", "Eps", "prop"] 

            if args.bab_kw: method_name += 'KW_' ; columns += ['BSAT_KW', 'BBran_KW', 'BTime_KW']
            if args.bab_gnn: method_name += 'GNN_'; columns += ['BSAT_gnnkwT', 'BBran_gnnkwT', 'BTime_gnnkwT']
            if args.bab_online: method_name += 'GnnOnline_'; columns += ['BSAT_gnnkwTO', 'BBran_gnnkwTO', 'BTime_gnnkwTO']
            if args.gurobi: method_name += 'GRB_'; columns += ['GSAT', 'GTime']

            record_name = result_path + f'{args.pdprops[:-4]}_{args.model_name}_{method_name}{args.cpu_id}.pkl'

        if os.path.isfile(record_name):
            graph_df = pd.read_pickle(record_name)
        else:
            indices= list(range(len(batch_ids)))
            

            graph_df = pd.DataFrame(index = indices, columns=columns)
            graph_df.to_pickle(record_name) 
    
    #skip = False

    for new_idx, idx in enumerate(batch_ids):
        # record_info 
        if args.record:
            graph_df = pd.read_pickle(record_name)
            if pd.isna(graph_df.loc[new_idx]['Eps'])==False:
                print(f'the {new_idx}th element is done')
                #skip = True
                continue
        #if skip == True:
        #    print(f'skip the {new_idx}th element')
        #    skip = False
        #    continue

        imag_idx = gt_results.loc[idx]["Idx"]
        prop_idx = gt_results.loc[idx]['prop']
        eps_temp = gt_results.loc[idx]["Eps"]        


        if args.data == 'cifar':
            x, verif_layers, test = load_cifar_1to1_exp(args.nn_name,imag_idx, prop_idx)
            # since we normalise cifar data set, it is unbounded now
            bounded = False
            assert test == prop_idx
            domain = torch.stack([x.squeeze(0) - eps_temp,x.squeeze(0) + eps_temp], dim=-1)
            linear = False
        else:
            raise NotImplementedError


        ### BaB
        if args.bab_kw:
            gt_prop = f'idx_{imag_idx}_prop_{prop_idx}_eps_{eps_temp}'
            print(gt_prop)
            kw_start = time.time()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = multiprocessing.Process(target = bab, args=(verif_layers, domain, x, eps_temp, 'kw', linear, None, bounded, return_dict))
            p.start()
            p.join(args.timeout)
            if p.is_alive():
                print("BaB KW Timeout")
                p.terminate()
                p.join()
                kw_min_lb = None; kw_min_ub = None; kw_ub_point = None; kw_nb_states= None
                kw_out="timeout"
            else:
                kw_min_lb = return_dict["min_lb"] 
                kw_min_ub = return_dict["min_ub"] 
                kw_ub_point = return_dict["ub_point"] 
                kw_nb_states = return_dict["nb_states"] 
                if kw_min_lb is None:
                    kw_out = 'grbError'
                else:
                    if kw_min_lb >= 0:
                        print("UNSAT")
                        kw_out = "False"
                    elif kw_min_ub < 0:
                        # Verify that it is a valid solution
                        print("SAT")
                        kw_out = "True"
                    else:
                        print("Unknown")
                        import pdb; pdb.set_trace()
                print(f"Nb states visited: {kw_nb_states}")
                #print('bnb takes: ', bnb_time)
                print('\n')
            #except KeyboardInterrupt:
            #    return
            #except:
            #    min_lb = None; min_ub = None; ub_point = None; nb_states= None
            #    out='Error'

            kw_end = time.time()
            kw_time = kw_end - kw_start
            print('total time required: ', kw_time)

            print('\n')





        ### BaB_gnn
        if args.bab_gnn:
            branching = 'graph'
            trace_name = f"{args.nn_name}_{args.model_name}_gnnkwT_{pref_branching_thd}_idx_{imag_idx}_prop_{prop_idx}_ball_{eps_temp}"
            print(trace_name)
            gnn_start = time.time()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            #try:
            p = multiprocessing.Process(target = bab_gnn, args=(verif_layers, domain, x, eps_temp, pref_branching_thd, bounded, branching, linear, models[args.model_name], trace_name, return_dict))
            p.start()
            p.join(args.timeout)
            if p.is_alive():
                print("BaB GNN Timeout")
                p.terminate()
                p.join()
                gnn_min_lb = None; gnn_min_ub = None; gnn_ub_point = None; gnn_nb_states= None
                gnn_out="timeout"
            else:
                gnn_min_lb = return_dict["min_lb"] 
                gnn_min_ub = return_dict["min_ub"] 
                gnn_ub_point = return_dict["ub_point"] 
                gnn_nb_states = return_dict["nb_states"] 
                if gnn_min_lb is None:
                    gnn_out = 'grbError'
                else:
                    if gnn_min_lb >= 0:
                        print("UNSAT")
                        gnn_out = "False"
                    elif gnn_min_ub < 0:
                        # Verify that it is a valid solution
                        print("SAT")
                        gnn_out = "True"
                    else:
                        print("Unknown")
                        import pdb; pdb.set_trace()
                print(f"Nb states visited: {gnn_nb_states}")
                #print('bnb takes: ', bnb_time)
                print('\n')
            #except KeyboardInterrupt:
            #    return
            #except:
            #    min_lb = None; min_ub = None; ub_point = None; nb_states= None
            #    out='Error'

            gnn_end = time.time()
            gnn_time = gnn_end - gnn_start
            print('total time required: ', gnn_time)

            print('\n')


        # ONLINE
        if args.bab_online:
            branching='graph'
            trace_name = f"{args.nn_name}_{args.model_name}_gnnkwTO_{pref_branching_thd}_online_{pref_online_thd}_idx_{imag_idx}_prop_{prop_idx}_ball_{eps_temp}"
            print(trace_name)
            print('Online')
            online_start = time.time()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            #try:
            p = multiprocessing.Process(target = bab_online, args=(verif_layers, domain, x, eps_temp, pref_branching_thd, pref_online_thd, bounded, branching, linear, models[args.model_name], trace_name, return_dict))
            p.start()
            p.join(args.timeout)
            if p.is_alive():
                print("BaB GnnOnline Timeout")
                p.terminate()
                p.join()
                online_min_lb = None; online_min_ub = None; online_ub_point = None; online_nb_states= None
                online_out="timeout"
            else:
                online_min_lb = return_dict["min_lb"] 
                online_min_ub = return_dict["min_ub"] 
                online_ub_point = return_dict["ub_point"] 
                online_nb_states = return_dict["nb_states"] 
                if online_min_lb is None:
                    online_out = 'grbError'
                else:
                    if online_min_lb >= 0:
                        print("UNSAT")
                        online_out = "False"
                    elif online_min_ub < 0:
                        # Verify that it is a valid solution
                        print("SAT")
                        online_out = "True"
                    else:
                        print("Unknown")
                        import pdb; pdb.set_trace()
                print(f"Nb states visited: {online_nb_states}")
                #print('bnb takes: ', bnb_time)
                print('\n')
            #except KeyboardInterrupt:
            #    return
            #except:
            #    min_lb = None; min_ub = None; ub_point = None; nb_states= None
            #    out='Error'

            online_end = time.time()
            online_time = online_end -online_start
            print('total time required: ', online_time)

            print('\n')

        if args.gurobi:
            guro_start = time.time()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            #try:
            p = multiprocessing.Process(target = gurobi, args=(verif_layers, domain, return_dict))
            p.start()
            p.join(args.timeout)
            if p.is_alive():
                print("gurobi Timeout")
                p.terminate()
                p.join()
                guro_out="timeout"
            else:
                guro_out  = return_dict["out"] 
                guro_nb_states = return_dict["nb_states"] 
            #except KeyboardInterrupt:
            #    return
            #except:
            #    min_lb = None; min_ub = None; ub_point = None; nb_states= None
            #    out='Error'

            guro_end = time.time()
            guro_time = guro_end - guro_start
            print('total time required: ', guro_time)
            print('results: ', guro_out)



        if args.record:  
            graph_df.loc[new_idx]["Idx"] = imag_idx 
            graph_df.loc[new_idx]["Eps"] = eps_temp
            graph_df.loc[new_idx]["prop"] = prop_idx

            if args.bab_kw:
                graph_df.loc[new_idx]["BSAT_KW"] = kw_out
                graph_df.loc[new_idx]["BBran_KW"] = kw_nb_states
                graph_df.loc[new_idx]["BTime_KW"] = kw_time

            if args.bab_gnn:
                graph_df.loc[new_idx]["BSAT_gnnkwT"] = gnn_out
                graph_df.loc[new_idx]["BBran_gnnkwT"] = gnn_nb_states
                graph_df.loc[new_idx]["BTime_gnnkwT"] = gnn_time

            if args.bab_online:
                graph_df.loc[new_idx]["BSAT_gnnkwTO"] = online_out
                graph_df.loc[new_idx]["BBran_gnnkwTO"] = online_nb_states
                graph_df.loc[new_idx]["BTime_gnnkwTO"] = online_time

            if args.gurobi:
                graph_df.loc[new_idx]["GSAT"] = guro_out
                graph_df.loc[new_idx]["GTime"] = guro_time

            graph_df.to_pickle(record_name) 



        
if __name__ == '__main__':
    main()
