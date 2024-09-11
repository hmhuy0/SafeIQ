import multiprocessing
import subprocess

dicts = {
    'seed': [0,1,2,3,4],
    'expert.n_bad': [100],
    'expert.n_mix_good': [
        100,
        # 400,
        # 1600,
        ],
    'expert.n_mix_bad': [1600],
    'env': ['car_goal'],
}

add_scripts =  [[
        'python' ,'-u', 'train_safeiq.py', 'agent=sac', 
        'method.loss=v0', 'method.chi=True', 'agent.init_temp=0.001', 
        'agent.actor_lr=1e-4', 'agent.disc_lr=1e-4', 'train.batch=256', 
        'env.eval_interval=10000', 'env.learn_steps=1e6',
        'agent.cql=False', 'agent.pen_bad=True', 'agent.disc_cfg.reward_factor=5.0',
        'env.clip_threshold=1', 'eval.eps=100',
        'wandb_log=True',
]]

for key in dicts.keys():
    new_add_scripts = []
    for x in dicts[key]:
        for script in add_scripts:
            new_add_scripts.append(script + [f'{key}={x}'])
    add_scripts = new_add_scripts



scripts = add_scripts

for s in scripts:
    print(s)

def run_manual_script(script):
    subprocess.run(script)

if __name__ == '__main__':
    max_processes = min(6,len(scripts))  # Number of scripts to run concurrently
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.map(run_manual_script, scripts)

    print("All scripts have finished running.")
