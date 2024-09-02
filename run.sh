export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hoang/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

CUDA_VISIBLE_DEVICES=2 python train_safeiq.py env=point_button agent=sac \
method.loss=no method.chi=True agent.init_temp=0.001 \
agent.actor_lr=3e-5 agent.disc_lr=1e-4 train.batch=512 \
env.eval_interval=5000 eval.eps=25 seed=0 env.learn_steps=3e6 \
agent.cql=False agent.pen_bad=True agent.disc_cfg.reward_factor=3.0 \
env.clip_threshold=0.75

# CUDA_VISIBLE_DEVICES=1 python train_bc.py env=point_button agent=sac \
# agent.actor_lr=1e-4 \
# env.eval_interval=5000 seed=0
