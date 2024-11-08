import os
import os.path as osp
import pathlib
import sys
import numpy as np
import time
import argparse
import yaml
import pickle
import random
import wandb
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax import jax_utils

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.videogpt.models import AE, VideoGPT
from viper_rl.videogpt.sampler import VideoGPTSampler
from viper_rl.videogpt.data import load_dataset
from viper_rl.videogpt.train_utils import init_model_state_videogpt, get_first_device, ProgressMeter, \
    save_video_grid, add_border, save_video
    
def print_class_methods(obj, include_private=False):
    """Print all methods of a class or object using different approaches.
    
    Args:
        obj: The class or object to inspect
        include_private: Whether to include private methods (starting with _)
    """
    # Get the class if an instance is passed
    cls = obj if isinstance(obj, type) else obj.__class__
    
    print(f"\n=== Methods of {cls.__name__} ===\n")
    
    # Method 1: Using dir()
    print("Using dir():")
    methods = []
    for attr in dir(obj):
        # Get the attribute
        try:
            attr_value = getattr(obj, attr)
        except:
            continue
            
        # Check if it's a method
        if callable(attr_value):
            if include_private or not attr.startswith('_'):
                methods.append(attr)
    
    for method in sorted(methods):
        print(f"  - {method}")
    
    # Method 2: Using inspect
    print("\nUsing inspect:")
    import inspect
    
    members = inspect.getmembers(obj, predicate=inspect.ismethod)
    for name, method in members:
        if include_private or not name.startswith('_'):
            try:
                signature = inspect.signature(method)
                print(f"  - {name}{signature}")
            except ValueError:
                print(f"  - {name}()")
    
    # Method 3: Get method docstrings
    print("\nMethods with docstrings:")
    for method_name in methods:
        try:
            method = getattr(obj, method_name)
            if method.__doc__:
                print(f"\n  {method_name}:")
                print(f"    {method.__doc__.strip()}")
        except:
            continue

    
def collect_data(agent, env, config, num_episodes=100, save_frequency=200):
    """Collect data with periodic saving to manage memory"""
    save_dir = Path(config.logdir) / 'collected_data'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    batch_idx = 0
    num_envs = len(env)
    episodes_per_env = num_episodes // num_envs
    remaining_episodes = num_episodes % num_envs

    total_episodes = [episodes_per_env] * num_envs
    for i in range(remaining_episodes):
        total_episodes[i] += 1

    env_episode_counts = [0] * num_envs
    env_done = [False] * num_envs
    
    total_collected_episodes = 0

    # Initial reset using DreamerV3's API - properly format action for batch env
    act_shape = env.act_space['action'].shape
    action_reset = {
        'action': jnp.zeros([num_envs, *act_shape]),
        'reset': jnp.ones(num_envs, bool)  # Array of True for all envs
    }
    # Convert inputs using JAXAgent's method
    action_reset = agent._convert_inps(action_reset, agent.policy_devices)
    obs = env.step(action_reset)
    obs = agent._convert_inps(obs, agent.policy_devices)
    
    # Let the policy handle the state
    state = None
    
    

    pbar = tqdm(total=num_episodes, desc="Collecting episodes")
    while sum(env_episode_counts) < num_episodes:
        # Get action in eval mode
        action, state = agent.policy(obs, state, mode='eval')
        action = jax.device_get(action)
        
        # Step environment with proper action format
        env_action = {
            'action': action['action'],
            'reset': jnp.zeros(num_envs, bool)  # Array of False for all envs
        }
        # Convert inputs using JAXAgent's method
        env_action = agent._convert_inps(env_action, agent.policy_devices)
        next_obs = env.step(env_action)
        next_obs = agent._convert_inps(next_obs, agent.policy_devices)


        # Convert to host memory for data collection
        next_obs_np = agent._convert_outs(next_obs, agent.policy_devices)
        obs_np = agent._convert_outs(obs, agent.policy_devices)
        action_np = agent._convert_outs(action, agent.policy_devices)

        for i in range(num_envs):
            if env_done[i]:
                continue
            
            episode_data = {
                'episode': env_episode_counts[i],
                'observation': {k: v[i] for k, v in obs_np.items()},
                'rgb_image': next_obs['image'][i],
                'action': action_np['action'][i],
                'reward': float(next_obs_np['reward'][i]),
                'done': bool(next_obs_np['is_last'][i]),
                'is_first': bool(next_obs_np['is_first'][i])
            }
            data.append(episode_data)

            
            if next_obs['is_last'][i]:
                env_done[i] = True
                env_episode_counts[i] += 1
                pbar.update(1)
                
                # Save batch if we've collected enough episodes
                if len(data) >= save_frequency:
                    save_batch_data(data, save_dir, batch_idx)
                    batch_idx += 1
                    data = []  # Clear the data list
                
                if env_episode_counts[i] < total_episodes[i]:
                    reset_action = {
                        'action': np.zeros([num_envs, *act_shape]),
                        'reset': np.zeros(num_envs, dtype=bool)
                    }
                    reset_action['reset'][i] = True
                    reset_action = agent._convert_inps(reset_action, agent.policy_devices)
                    obs = env.step(reset_action)
                    obs = agent._convert_inps(obs, agent.policy_devices)
                    state = None
                    env_done[i] = False
                    
        obs = next_obs
    pbar.close()

    # Save any remaining data
    if data:
        save_batch_data(data, save_dir, batch_idx)
        batch_idx += 1

    metadata = {
        'total_episodes': num_episodes,
        'total_batches': batch_idx,
        'save_frequency': save_frequency,
        'observation_space': {k: v.shape for k, v in obs_np.items()},
        'action_shape': action_np['action'].shape
    }
    np.save(save_dir / 'metadata.npy', metadata)
    
    print(f"Data saved in {batch_idx} batches to {save_dir}")
    return None

def load_collected_data(data_dir, batch_indices=None):
    """Load collected data from batches"""
    data_dir = Path(data_dir)
    metadata = np.load(data_dir / 'metadata.npy', allow_pickle=True).item()
    
    if batch_indices is None:
        batch_indices = range(metadata['total_batches'])
        
    all_data = []
    for i in batch_indices:
        batch_dir = data_dir / f'batch_{i}'
        arrays = np.load(batch_dir / 'trajectory_data.npz')
        all_data.append(arrays)
        
    return all_data, metadata

def save_batch_data(processed_data, save_dir, batch_idx):
    """Helper function to save a batch of episodes with all data using only numpy arrays"""
    # Create batch directory
    batch_dir = save_dir / f'batch_{batch_idx}'
    batch_dir.mkdir(exist_ok=True)
    
    # Create arrays for all data
    array_data = {
        'episodes': np.array([d['episode'] for d in processed_data]),
        'rgb_images': np.array([d['rgb_image'] for d in processed_data]),
        'actions': np.array([d['action'] for d in processed_data]),
        'rewards': np.array([d['reward'] for d in processed_data]),
        'dones': np.array([d['done'] for d in processed_data]),
        'is_firsts': np.array([d['is_first'] for d in processed_data])
    }
    
    # Add observation arrays
    if processed_data and 'observation' in processed_data[0]:
        obs_dict = processed_data[0]['observation']
        for key in obs_dict.keys():
            array_data[f'obs_{key}'] = np.array([
                d['observation'][key] for d in processed_data
            ])
    
    # Save everything in a single npz file
    np.savez_compressed(batch_dir / 'trajectory_data.npz', **array_data)    
    
def main():
    global model
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    config.ckpt = config.output_dir if osp.exists(config.output_dir) else None

    if is_master_process:
        wandb.init(project='viper_rl', config=config,
                   id=config.run_id, resume='allow', mode='online')
        wandb.run.name = config.run_id
        wandb.run.save()

    train_loader, class_map, _ = load_dataset(config, train=True, modality='video')
    test_loader, class_map_test, _ = load_dataset(config, train=False, modality='video')

    if config.class_cond:
        assert class_map == class_map_test, (class_map, class_map_test)
        pickle.dump(class_map, open(osp.join(config.output_dir, 'class_map.pkl'), 'wb'))

    ae = AE(config.ae_ckpt)

    batch = next(train_loader)
    batch = ae.prepare_batch(batch)
    batch = get_first_device(batch)

    model = VideoGPT(config, ae)
    sampler = VideoGPTSampler(model)
    state, schedule_fn = init_model_state_videogpt(init_rng, model, batch, config)

    if config.ckpt is not None:
        state = checkpoints.restore_checkpoint(osp.join(config.ckpt, 'checkpoints'), state)
        print(f'Restored from checkpoint {osp.join(config.ckpt)}, at itr {int(state.step)}')
    
    iteration = int(state.step)
    state = jax_utils.replicate(state)

    ckpt_dir = osp.join(config.output_dir, 'checkpoints')
    ckpt_dir = os.path.abspath(ckpt_dir)  # Convert to absolute path

    rng = jax.random.fold_in(rng, jax.process_index() + random.randint(0, 100000))
    rngs = jax.random.split(rng, jax.local_device_count())
    best_loss = float('inf')
    best_state = None
    while iteration <= config.total_steps:
        iteration, state, rngs = train(iteration, ae, model, state, train_loader,
                                       schedule_fn, rngs)
        if iteration % config.test_interval == 0:
            val_loss, rngs = validate(iteration, ae, model, state, test_loader, rngs)
            is_best = val_loss < best_loss
            if is_best:
                best_state = jax_utils.unreplicate(state)
                best_loss = min(best_loss, val_loss)
        if iteration % config.viz_interval == 0:
            visualize(sampler, ae, iteration, state, test_loader)
        if iteration % config.save_interval == 0 and is_master_process and is_best:
            state_ = jax_utils.unreplicate(state)
            save_path = checkpoints.save_checkpoint(ckpt_dir, state_, state_.step, keep=1, overwrite=True)
            print('Saved checkpoint to', save_path)
            del state_
        iteration += 1
        
    print("Training complete. Collecting data...")


def train_step(batch, state, rng):
    def loss_fn(params, batch, rng):
        rng_dropout, new_rng = jax.random.split(rng)
        variables = {'params': params}
        out = state.apply_fn(
            variables,
            **batch,
            training=True,
            rngs={'dropout': rng_dropout},
            method=model.loss
        )
        return out['loss'], (out, new_rng)
    aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch, rng)
    out, rng = aux[1]
    grads = jax.lax.pmean(grads, axis_name='device')
    new_state = state.apply_gradients(
        grads=grads,
    )

    if config.ema:
        decay = jnp.where(state.step == 0, 0.0, config.ema)
        ema_params = jax.tree_util.tree_map(
            lambda a, b: decay * a + (1.0 - decay) * b,
            state.ema_params, new_state.params
        )
        new_state = new_state.replace(ema_params=ema_params)
    return new_state, out, rng


def train(iteration, ae, model, state, train_loader, schedule_fn, rngs):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + model.metrics
    )

    p_train_step = jax.pmap(train_step, axis_name='device', donate_argnums=(0, 1, 2))

    end = time.time()
    while True:
        batch = next(train_loader)
        batch_size = batch[list(batch.keys())[0]].shape[1]
        progress.update(data=time.time() - end)

        batch = ae.prepare_batch(batch)
        state, return_dict, rngs = p_train_step(batch=batch, state=state, rng=rngs)

        metrics = {k: return_dict[k].mean() for k in model.metrics}
        metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process:
            wandb.log({'train/lr': jax.device_get(schedule_fn(iteration))}, step=iteration)
            wandb.log(jax.device_get({**{f'train/{metric}': val
                        for metric, val in metrics.items()}
                    }), step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        if iteration % config.viz_interval == 0 or \
        iteration % config.test_interval == 0 or \
        iteration % config.save_interval == 0 or \
        iteration >= config.total_steps:
            return iteration, state, rngs

        iteration += 1


def val_step(batch, state, rng):
    def loss_fn(params, batch, rng):
        rng_dropout, new_rng = jax.random.split(rng)
        variables = {'params': params}
        out = state.apply_fn(
            variables,
            **batch,
            training=False,
            rngs={'dropout': rng_dropout},
            method=model.loss
        )
        return out, new_rng
    out, rng = loss_fn(state.params, batch, rng)
    out = jax.lax.pmean(out, axis_name='device')
    return out, rng


def validate(iteration, ae, model, state, test_loader, rngs):
    progress = ProgressMeter(
        50,
        ['time', 'data'] + model.metrics,
        prefix='\tTest:'
    )

    p_val_step = jax.pmap(val_step, axis_name='device', donate_argnums=(0, 1, 2))

    end = time.time()
    for i in range(50):
        batch = next(test_loader)
        batch_size = batch[list(batch.keys())[0]].shape[1]
        progress.update(data=time.time() - end)

        batch = ae.prepare_batch(batch)
        return_dict, rngs = p_val_step(batch=batch, state=state, rng=rngs)

        metrics = {k: return_dict[k].mean() for k in model.metrics}
        metrics = jax.device_get({k: v.astype(jnp.float32) for k, v in metrics.items()})
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})
        progress.update(time=time.time() - end)
        end = time.time()

        if i % config.log_interval == 0:
            progress.display(i)

    progress.display(i)

    metrics = {metric: progress.meters[metric].avg
               for metric in model.metrics}

    if is_master_process:
        wandb.log({**{f'val/{metric}': val
                      for metric, val in metrics.items()}
                  }, step=iteration)
    return metrics['loss'], rngs


def visualize(sampler, ae, iteration, state, test_loader):
    batch = next(test_loader)
    video = batch['video']
    if len(video.shape) == 5: # NBTHW
        video = ae.decode(video)
    variables = {'params': state.ema_params if hasattr(state, 'ema_params') else state.params}
    samples = sampler(variables, batch).copy()
    samples = samples.reshape(-1, *samples.shape[-4:])
    real = jax.device_get(video)
    real = (real * 0.5 + 0.5).reshape(-1, *real.shape[-4:])
    add_border(samples[:, :config.open_loop_ctx], (0., 1., 0.))
    add_border(samples[:, config.open_loop_ctx:], (1., 0., 0.))

    videos = np.stack((samples, real), axis=1)
    videos = videos.reshape(-1, *videos.shape[2:])
    videos = (videos * 255).astype(np.uint8)

    videos = save_video_grid(videos)
    if is_master_process:
        videos = np.transpose(videos, (0, 3, 1, 2))
        wandb.log({'viz/sample': wandb.Video(videos, format='gif')}, step=iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    args.run_id = args.output_dir.split('/')[-1] + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX total devices: {jax.device_count()}')
    print(f'JAX local devices: {jax.local_device_count()}')

    config = yaml.safe_load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['viz_interval'] = 10
        config['log_interval'] = 1
        config['test_interval'] = 10
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'

    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args

    is_master_process = jax.process_index() == 0

    main()
