import subprocess

dicts = {
    'expert.n_bad': [100],
    'expert.n_mix_good': [
        # 100,
        100,
        # 1600,
        ],
    'expert.n_mix_bad': [100],
    'seed': [0],
    'env': [
        # 'point_goal',
        # 'point_button',
        # 'car_goal',
        # 'car_button',
        'cheetah_vel',
            ],
}

# add_scripts =  [[
#         'python' ,'-u', 'train_bc.py', 'agent=sac', 
#         'method.loss=v0', 'method.chi=True', 'agent.init_temp=0.001', 
#         'agent.actor_lr=1e-4', 'agent.disc_lr=1e-4', 'train.batch=256', 
#         'env.eval_interval=10000', 'env.learn_steps=1e6',
#         'env.clip_threshold=1', 'eval.eps=100',
#         'wandb_log=True',
# ]]

add_scripts =  [[
        'python' ,'-u', 'train_safeiq_velocity.py', 'agent=sac', 
        'method.loss=v0', 'method.chi=True', 'agent.init_temp=0.001',         
        'agent.actor_lr=1e-4', 'agent.disc_lr=1e-4', 'train.batch=256', 
        'env.eval_interval=10000', 'env.learn_steps=1e6',
        'agent.cql=False', 'agent.pen_bad=True', 'agent.disc_cfg.reward_factor=5.0',
        'env.clip_threshold=1', 'eval.eps=100',
        'train.update_actor_Q=False','train.sep_V=False',

        'agent.disc_cfg.hidden_depth=2','diag_gaussian_actor.hidden_depth=2','q_net.hidden_depth=2',
        'wandb_log=False',
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

from concurrent.futures import ThreadPoolExecutor, as_completed



if __name__ == '__main__':
    max_processes = min(5,len(scripts))  # Number of scripts to run concurrently
    # run script every 5 seconds
    import time
    print('\n'*5)
    print('start running')    
    
    with ThreadPoolExecutor(max_workers=max_processes) as executor:
        future_to_script = {executor.submit(run_manual_script, script): script for script in scripts}
        
        for future in as_completed(future_to_script):
            script_name = future_to_script[future]



