# -*- coding: future_fstrings -*-
import sys, os, time

t0 = time.time()
import socket
import numpy as np
import torch
from ruamel.yaml import YAML
from absl import flags
from utils import system, logger
from pathlib import Path
import psutil

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner

FLAGS = flags.FLAGS
flags.DEFINE_string("cfg", None, "path to configuration file")
flags.DEFINE_string("env", None, "env_name")
flags.DEFINE_string("algo", None, '["td3", "sac"]')

flags.DEFINE_boolean("automatic_entropy_tuning", None, "for [sac]")
flags.DEFINE_float("target_entropy", None, "for [sac]")
flags.DEFINE_float("entropy_alpha", None, "for [sac]")
flags.DEFINE_float("ac_lr", None, "actor critic lr")
flags.DEFINE_float("time_limit", 1000.0, "time limit (discovery)")
flags.DEFINE_float("actor_aux_weight", None, "aux actor weight")
flags.DEFINE_float("critic_aux_weight", None, "aux critic weight")
flags.DEFINE_float("init_alpha", None, "init value for alpha")

flags.DEFINE_integer("seed", None, "seed")
flags.DEFINE_integer("batch_size", None, "batch_size")
flags.DEFINE_integer("save_interval", None, "save_interval")
flags.DEFINE_integer("num_init_episodes", None, "num init episode used")
flags.DEFINE_integer("num_iters", None, "num episodes trained")
flags.DEFINE_integer("cuda", None, "cuda device id")
flags.DEFINE_string("policy_dir", None, "directory to the policy folder")
flags.DEFINE_string("buffer_type", None,
                    "buffer [seq_vanilla, seq_per]")
flags.DEFINE_string("checkpoint_dir", None, "directory for checkpoint")
flags.DEFINE_string("actor_type", None, "type of actor [normal, equi, aug-aux, aug, aux]")
flags.DEFINE_string("critic_type", None, "type of critic [normal, equi, aug-aux, aug, aux]")
flags.DEFINE_string("group_name", None, "type of group used")
flags.DEFINE_string("prefix", None, "prefix of wandb group")
flags.DEFINE_boolean("replay", False, "replay/train mode")
flags.DEFINE_boolean("explore_deterministic", True, "choosing explore stochastic/deterministic")
flags.DEFINE_boolean("save_rollouts", False, "save rollouts from an agent")

flags.FLAGS(sys.argv)
yaml = YAML()
v = yaml.load(open(FLAGS.cfg))

# overwrite config params
if FLAGS.env is not None:
    v["env"]["env_name"] = FLAGS.env

if FLAGS.algo is not None:
    v["policy"]["algo_name"] = FLAGS.algo

if FLAGS.num_init_episodes is not None:
    v["train"]["num_init_rollouts_pool"] = FLAGS.num_init_episodes

if FLAGS.num_iters is not None:
    v["train"]["num_iters"] = FLAGS.num_iters

if FLAGS.buffer_type is not None:
    v["train"]["buffer_type"] = FLAGS.buffer_type

if FLAGS.batch_size is not None:
    v["train"]["batch_size"] = FLAGS.batch_size

if FLAGS.actor_type is not None:
    v["policy"]["actor_type"] = FLAGS.actor_type

if FLAGS.critic_type is not None:
    v["policy"]["critic_type"] = FLAGS.critic_type

if FLAGS.actor_aux_weight is not None:
    v["policy"]["actor_aux_weight"] = FLAGS.actor_aux_weight

if FLAGS.critic_aux_weight is not None:
    v["policy"]["critic_aux_weight"] = FLAGS.critic_aux_weight

if FLAGS.ac_lr is not None:
    v["policy"]["lr"] = FLAGS.ac_lr

if FLAGS.group_name is not None:
    v["policy"]["group_name"] = FLAGS.group_name

if FLAGS.save_interval is not None:
    v["eval"]["save_interval"] = FLAGS.save_interval

actor_type, critic_type = v["policy"]["actor_type"], v["policy"]["critic_type"]

assert actor_type in ["normal", "equi", "mlp", "aug-aux", "aug", "aux"]
assert critic_type in ["normal", "equi", "mlp", "aug-aux", "aug", "aux"]

algo = v["policy"]["algo_name"]

assert algo in ["sac", "td3"]

if FLAGS.init_alpha is not None:
    v["policy"][algo]["init_alpha"] = FLAGS.init_alpha

if FLAGS.automatic_entropy_tuning is not None:
    v["policy"][algo]["automatic_entropy_tuning"] = FLAGS.automatic_entropy_tuning
if FLAGS.entropy_alpha is not None:
    v["policy"][algo]["entropy_alpha"] = FLAGS.entropy_alpha
if FLAGS.target_entropy is not None:
    v["policy"][algo]["target_entropy"] = FLAGS.target_entropy

if FLAGS.seed is not None:
    v["seed"] = FLAGS.seed

if socket.gethostname() in ['theseus', 'titan']:
    if FLAGS.cuda is not None:
        v["cuda"] = FLAGS.cuda
else:
    v["cuda"] = 0

# system: device, threads, seed, pid
seed = v["seed"]
system.reproduce(seed)

torch.set_num_threads(1)
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    pid += "_" + str(os.environ["SLURM_JOB_ID"])  # use job id

# set gpu
set_gpu_mode(torch.cuda.is_available() and v["cuda"] >= 0, v["cuda"])

env_name = v["env"]["env_name"]
exp_id = f"logs/{env_name}/"

exp_id += f"{algo}_{actor_type}_{critic_type}_" + \
          f"e{v['train']['use_expert_rollouts']}_"

if 'equi' in actor_type or 'equi' in critic_type:
    exp_id += f"rot{v['policy']['group_name']}_"

if algo in ["sac", "sacfd"]:
    if not v["policy"][algo]["automatic_entropy_tuning"]:
        exp_id += f"alpha-{v['policy'][algo]['entropy_alpha']}_"
    elif "target_entropy" in v["policy"]:
        exp_id += f"ent-{v['policy'][algo]['target_entropy']}_"

exp_id += f"gamma-{v['policy']['gamma']}_"

if actor_type != "mlp":
    exp_id += f"len-{v['train']['sampled_seq_len']}_bs-{v['train']['batch_size']}_"
    exp_id += f"freq-{v['train']['num_updates_per_iter']}_"
    policy_input_str = "o"
    if v["policy"]["action_embedding_size"] > 0:
        policy_input_str += "a"
    exp_id += policy_input_str + "/"

exp_id += 'seed-' + str(seed) + "/"

os.makedirs(exp_id, exist_ok=True)
log_folder = os.path.join(exp_id, system.now_str())
logger_formats = ["stdout", "log", "csv"]
if v["eval"]["log_tensorboard"]:
    logger_formats.append("tensorboard")
logger.configure(dir=log_folder, format_strs=logger_formats, precision=4)
logger.log(f"preload cost {time.time() - t0:.2f}s")

# os.system(f"cp -r policies/ {log_folder}")
yaml.dump(v, Path(f"{log_folder}/variant_{pid}.yml"))
key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
logger.log("\n".join(f.serialize() for f in key_flags) + "\n")
logger.log("pid", pid, socket.gethostname())
os.makedirs(os.path.join(logger.get_dir(), "save"))


# start training
learner = Learner(
    env_args=v["env"],
    train_args=v["train"],
    eval_args=v["eval"],
    policy_args=v["policy"],
    seed=seed,
    replay=FLAGS.replay,
    time_limit=FLAGS.time_limit,
    explore_deterministic=FLAGS.explore_deterministic,
    prefix=FLAGS.prefix,
    ckpt_dir=FLAGS.checkpoint_dir,
    replay_policy=FLAGS.replay,
    cfg_file=FLAGS.cfg,
)

logger.log(
    f"total RAM usage: {psutil.Process().memory_info().rss / 1024 ** 3 :.2f} GB\n"
)

if FLAGS.save_rollouts:
    assert FLAGS.policy_dir is not None
    learner.save_rollouts(FLAGS.policy_dir)
elif FLAGS.replay:
    assert FLAGS.policy_dir is not None
    learner.replay_policy(FLAGS.policy_dir)    
else:
    learner.train(FLAGS.policy_dir)
