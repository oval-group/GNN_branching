# GNN_branching
This repository contains all the code necessary to replicate the experiments
reported in the paper: [Neural Network Branching for Neural Network Verification](). 



## Structure of the repository

## Running the code
### Dependencies
The code was implemented assuming to be run under `python3.6`.
We have a dependency on:
* [The Gurobi solver](http://www.gurobi.com/) to solve the LP arising from the
Network linear approximation and the Integer programs for the MIP formulation.
Gurobi can be obtained
from [here](http://www.gurobi.com/downloads/gurobi-optimizer) and academic
licenses are available
from [here](http://www.gurobi.com/academia/for-universities).
* [Pytorch](http://pytorch.org/) to represent the Neural networks and to use as
  a Tensor library. 
  
### Installing everything
We recommend installing everything into a python virtual environment.

```bash
git clone --recursive https://github.com/oval-group/GNN_branching.git

cd PLNN-verification
virtualenv -p python3.6 ./venv
./venv/bin/activate

# Install gurobipy to this virtualenv
# (assuming your gurobi install is in /opt/gurobi701/linux64)
cd /opt/gurobi701/linux64/
python setup.py install
cd -

# Install pytorch to this virtualenv
# (or check updated install instructions at http://pytorch.org)
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 

# Install the code of this repository
python setup.py install

# Install the code for computing fast heuristic bounds
cd convex_adversarial
python setup.py install
```

### Running the experiments
If you have setup everything according to the previous instructions, you should
be able to replicate the experiments of the paper. To do so, follow the
following instructions:

```bash

## Generate the results
./scripts/bab_mip.sh

```
  

## Reference
If you use it in your research, please cite:

```
@Article{Lu2019,
  author        = {Lu, Jingyue and Kumar, M Pawan},
  title        =  {Neural Network Branching for Neural Network Verification},
  journal      = {},
  year         = {2019},
}
```
