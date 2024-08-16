export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hoang/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

CUDA_VISIBLE_DEVICES=0 python train_safeiq.py env=point_button agent=sac \
method.loss=v0 method.chi=True agent.actor_lr=3e-5 agent.init_temp=0.001 \
agent.disc_lr=1e-4 env.eval_interval=5000 seed=0 \
agent.cql=True

# CUDA_VISIBLE_DEVICES=1 python train_bc.py env=point_button agent=sac \
# agent.actor_lr=1e-4 \
# env.eval_interval=5000 seed=0
