## Symmetry-aware Reinforcement Learning for Robotic Assembly under Partial Observability with a Soft Wrist (ICRA-24)

<div style="text-align: center;">
  <a href="https://arxiv.org/abs/2402.18002">ArXiv Paper</a> | <a href="https://www.youtube.com/watch?v=XU4Sbt_NnT8">Submission Video</a>
</div>

## Abstract
This study tackles the representative yet challenging contact-rich peg-in-hole task of robotic assembly, using a soft wrist that can operate more safely and tolerate lower-frequency control signals than a rigid one. Previous studies often use a fully observable formulation, requiring external setups or estimators for the peg-to-hole pose. In contrast, we use a partially observable formulation and deep reinforcement learning from demonstrations to learn a memory-based agent that acts purely on haptic and proprioceptive signals. Moreover, previous works do not incorporate potential domain symmetry and thus must search for solutions in a bigger space. Instead, we propose to leverage the symmetry for sample efficiency by augmenting the training data and constructing auxiliary losses to force the agent to adhere to the symmetry. Results in simulation with five different symmetric peg shapes show that our proposed agent can be comparable to or even outperform a state-based agent. In particular, the sample efficiency also allows us to learn directly on the real robot within 3 hours. 


## Setup
1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. Create and activate environment
```
conda create --name symm_pomdp python=3.8.16
conda activate symm_pomdp
```
3. Clone this repository and install required packages
```
git clone https://github.com/hai-h-nguyen/symmetry-aware-pomdps.git
pip install -r requirements.txt
```
4. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) (we used 1.12.0 for cuda 10.2 but other versions should work)
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

## Generate demonstrations
Demonstration files for tasks are already stored in /demonstrations. If you want to generate new demontration files, use a joystick (we used a [Logitech G F310](https://www.amazon.com/dp/B003VAHYQY?psc=1&ref=ppx_yo2ov_dt_b_product_details)) and run the script:
```
python pomdp-domains/scripts/generate_human_demonstrations.py
```

## Train

```
export PYTHONPATH=${PWD}:$PYTHONPATH

# RSAC-Normal:
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-XYZ-v0 --seed 0 --cuda 0

# SAC-Obs:
python3 policies/main.py --cfg configs/peg_insertion/mlp.yml --env PegInsertion-Square-XYZ-v0 --seed 0 --cuda 0

# RSAC-Aug:
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-XYZ-v0 --seed 0 --cuda 0 --group_name FlipXY --actor_type aug --critic_type aug

# RSAC-Aug-Aux:
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-XYZ-v0 --seed 0 --cuda 0 --group_name FlipRotXY4 --actor_type aug-aux --critic_type aug-aux

# RSAC-Equi:
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-XYZ-v0 --seed 0 --cuda 0 --group_name FlipXY --actor_type equi --critic_type equi

# SAC-State:
python3 policies/main.py --cfg configs/peg_insertion/mlp.yml --env PegInsertion-Square-State-XYZ-v0 --seed 0 --cuda 0

```

## Simulate a Trained Policy
```
python3 policies/main.py --cfg configs/peg_insertion/rnn.yml --env PegInsertion-Square-XYZ-v0 --group_name FlipRotXY4 --actor_type aug-aux --critic_type aug-aux --replay --policy_dir policy.pt
```

## Peg & Hole Design File
```
See peg_and_hole_designs.f3d (AutoDesk project file), from which, mesh files for pegs and holes can be exported
```

## Acknowledgments
This repository is based on [pomdp-baselines](https://github.com/twni2016/pomdp-baselines) and [robosuite](https://github.com/ARISE-Initiative/robosuite).

## Citation
If you find our work helpful to your research, please cite us as
```
@article{nguyen2024symmetry,
  title={Symmetry-aware Reinforcement Learning for Robotic Assembly under Partial Observability with a Soft Wrist},
  author={Nguyen, Hai and Kozuno, Tadashi and Beltran-Hernandez, Cristian C and Hamaya, Masashi},
  journal={arXiv preprint arXiv:2402.18002},
  year={2024}
}
```
