# Symmetry-aware Reinforcement Learning for Robotic Assembly under Partial Observability with a Soft Wrist

## Setup
1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. Create and activate environment
```
conda create --env symm_pomdp python=3.8.16
conda activate symm_pomdp
```
3. Clone this repository and install required packages
```
git clone https://github.com/hai-h-nguyen/symmetry-aware-pomdps.git
pip install -r requirements.txt
```
4. Install [Pytorch](https://pytorch.org/get-started/previous-versions/) (we used 1.12.0 for cuda 10.2 but other versions should work)
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
```
5. Install submodules
```
cd robotsuite
pip install -r requirements.txt
pip install -e .
cd ..
cd pomdp-domains
pip install -e .
cd ..
```

---

## Train

```
export PYTHONPATH=${PWD}:$PYTHONPATH

# RSAC-Normal:
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-Old-XYZ-v0 --seed 0 --cuda 0

# SAC-Obs:
python3 policies/main.py --cfg configs/peg_insertion/mlp.yml --env PegInsertion-Square-Old-XYZ-v0 --seed 0 --cuda 0

# RSAC-Aug:
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-Old-XYZ-v0 --seed 0 --cuda 0 --group_name FlipXY --actor_type aug --critic_type aug

# RSAC-Aug-Aux:
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-Old-XYZ-v0 --seed 0 --cuda 0 --group_name FlipRotXY4 --actor_type aug-aux --critic_type aug-aux

# RSAC-Equi:
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-Old-XYZ-v0 --seed 0 --cuda 0 --group_name FlipXY --actor_type equi --critic_type equi

# SAC-State:
python3 policies/main.py --cfg configs/peg_insertion/mlp.yml --env PegInsertion-Square-Old-State-XYZ-v0 --seed 0 --cuda 0

```

## Simulate a Trained Policy
```
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-Old-XYZ-v0 --group_name FlipRotXY4 --actor_type aug-aux --critic_type aug-aux --replay --policy_dir policy.pt
```

## Peg & Hole Design File
```
See peg_and_hole_designs.f3d (AutoDesk project file), from which, mesh files for pegs and holes can be exported
```

