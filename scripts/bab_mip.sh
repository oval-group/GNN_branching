#!/bin/bash

# total number of cpu cores that will be used
cpus_total=1
# absolute cpu number (cpu index on the machine, based on htop command)
cpu_abs=0
# the relative task number, ranging from 0 to $cpus_total-1
task_no=0

# base model
timeout=3600
pdprops="base_easy.pkl"
#pdprops="base_med.pkl"
#pdprops="base_hard.pkl"
nn_name="cifar_base_kw"

## wide model
#timeout=7200
#pdprops="wide.pkl"
#nn_name="cifar_wide_kw"

## deep model
#timeout=7200
#pdprops="deep.pkl"
#nn_name="cifar_deep_kw"

# method
para="--bab_kw"
#para="--bab_gnn"
#para="--bab_online"
#para="--gurobi"

echo "taskset --cpu-list $cpu_abs python experiments/bab_mip.py --cpu_id $task_no --timeout $timeout --cpus_total $cpus_total --pdprops $pdprops $para --nn_name $nn_name --record"
taskset --cpu-list $cpu_abs python experiments/bab_mip.py --cpu_id $task_no --timeout $timeout --cpus_total $cpus_total --pdprops $pdprops $para --nn_name $nn_name --record 



