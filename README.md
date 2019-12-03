# Neural Network Branching for Neural Network Verification
This repository contains all the code necessary to replicate the experiments
reported in the paper: [Neural Network Branching for Neural Network Verification](). 

## Dependences
* This code should work for python >= 3.6 and pytorch >= 0.4.1.
* The commercial solver [Gurobi](http://www.gurobi.com/) is required for solving LPs arising from the
Network linear approximation and the Integer programs for the MIP formulation.
Gurobi can be obtained
from [here](http://www.gurobi.com/downloads/gurobi-optimizer) and academic
licenses are available
from [here](http://www.gurobi.com/academia/for-universities).
* We have included a modified version of the github package [convex_adversarial](https://github.com/locuslab/convex_adversarial) in the folder ./convex_adversaria/. The github package is used for computing intermediate bounds, which are needed in building LPs for Network linear approximations. We modify the original version ./convex_adversarial slightly to best accomodate our needs.
* The ./plnn/ is developed on the original implementations of Branch and Bound methods, provided in the github package [PLNN_verification](https://github.com/oval-group/PLNN-verification). We have also directly used the MIPplanet solver provided in  [PLNN_verification](https://github.com/oval-group/PLNN-verification).
  
## Installation
We recommend installing everything into a virtual environment.

```bash
git clone --recursive https://github.com/oval-group/GNN_branching.git

cd GNN_branching
virtualenv -p python3.6 ./gnn
./gnn/bin/activate

# Install gurobipy to this virtualenv
# (assuming your gurobi install is in /opt/gurobi801/linux64)
cd /opt/gurobi801/linux64/
python setup.py install
cd -

# Install pytorch to this virtualenv
# (check updated install instructions at http://pytorch.org)


# Install the code of this repository
python setup.py install

# Install the code for computing fast intermediate bounds
cd convex_adversarial
python setup.py install
```

## Running the experiments
* All verification properties with previous experimental results are recorded in the format of pandas pickle tables. Tables can be found in the folder ./cifar_exp/. For the base model, verification properties are divided into base_easy.pkl, base_med.pkl and base_hard.pkl according to BaBSR (bab_kw flag in the code) solving time. Wide.pkl and deep.pkl are properties for the wide and the deep model respectively.
* To reproduce the experiments for the base model, please run the bash script bab_mip.sh with the following code. 

```bash
## Generate the results
./scripts/bab_mip.sh

```
* Results are saved in pandas table as well in the newly created folder ./cifar_results/.
* For the wide and the deep model, please comment and uncomment out related parts in bab_mip.sh.
* In our experiments, we run the same method for all properties then move to the next method instead of running all methods for a property then moving to the next property.

  

## Reference
If you use this work in your research, please cite:

```
@Article{Lu2019,
  author        = {Lu, Jingyue and Kumar, M Pawan},
  title        =  {Neural Network Branching for Neural Network Verification},
  journal      = {},
  year         = {2019},
}
```
