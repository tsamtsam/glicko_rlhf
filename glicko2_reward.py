"""
Glicko-2 Reward Function Script for Berkeley-NEST Nectar Dataset
Where reward = rating - (rd_scale * ratings_deviation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle
import argparse
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm


class Glicko2:
    """
    Implementation of the Glicko-2 rating system.
    
    Glicko-2 is a method for assessing a player's strength in games of skill,
    such as chess. It expands upon the original Glicko rating system by adding
    a ratings volatility parameter.
    """
    
    def __init__(self, rating=1500, rd=350, vol=0.06, tau=0.5, epsilon=0.000001):
        """
        Initialize a Glicko-2 rating object.
        
        Parameters:
        rating (float): The rating parameter. Default is 1500.
        rd (float): The rating deviation parameter. Default is 350.
        vol (float): The volatility parameter. Default is 0.06.
        tau (float): System constant, constrains volatility. Default is 0.5.
        epsilon (float): Convergence tolerance. Default is 0.000001.
        """
        self.rating = rating
        self.rd = rd
        self.vol = vol
        self.tau = tau
        self.epsilon = epsilon
        self._scale_factor = 173.7178
    
    def g(self, rd):
        """
        This function computes the g(RD) function of the Glicko-2 system.
        
        Parameters:
        rd (float): Rating deviation.
        
        Returns:
        float: g(RD) value.
        """
        return 1 / np.sqrt(1 + (3 * rd**2) / (np.pi**2))
    
    def E(self, rating, other_rating, other_rd):
        """
        This function computes the expected score of a player against another player.
        
        Parameters:
        rating (float): Rating of the player.
        other_rating (float): Rating of the opponent.
        other_rd (float): Rating deviation of the opponent.
        
        Returns:
        float: Expected score (between 0 and 1).
        """
        return 1 / (1 + np.exp(-self.g(other_rd) * (rating - other_rating) / 400))
    
    def _volatility(self, rating, rd, vol, delta, v_inv):
        """
        This function computes the new volatility of the Glicko-2 system.
        
        Parameters:
        rating (float): Rating.
        rd (float): Rating deviation.
        vol (float): Volatility.
        delta (float): Delta value.
        v_inv (float): Inverse of variance.
        
        Returns:
        float: Updated volatility.
        """
        a = np.log(vol**2)
        tau = self.tau
        epsilon = self.epsilon
        
        def f(x):
            ex = np.exp(x)
            
            first_part = ex * (delta**2 - rd**2 - v_inv - ex) / (2 * (rd**2 + v_inv + ex)**2)
            second_part = (x - a) / (tau**2)
            
            return first_part - second_part
            
        # Initial values for iteration
        A = a
        B = 0
        
        if delta**2 > rd**2 + v_inv:
            B = np.log(delta**2 - rd**2 - v_inv)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau
            
        # Iteration to find the root of f(x)
        fa = f(A)
        fb = f(B)
        
        while abs(B - A) > epsilon:
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            
            if fc * fb < 0:
                A = B
                fa = fb
            else:
                fa = fa / 2
                
            B = C
            fb = fc
            
        return np.exp(A / 2)
    
    def update_player(self, rating, rd, vol, outcomes, other_ratings, other_rds):
        """
        This function computes the new rating, RD, and volatility of a player.
        
        Parameters:
        rating (float): Current rating.
        rd (float): Current rating deviation.
        vol (float): Current volatility.
        outcomes (list): List of game outcomes (1 for win, 0.5 for draw, 0 for loss).
        other_ratings (list): List of opponent ratings.
        other_rds (list): List of opponent rating deviations.
        
        Returns:
        tuple: (new_rating, new_rd, new_vol)
        """
        # Convert to Glicko-2 scale
        rating = (rating - 1500) / self._scale_factor
        rd = rd / self._scale_factor
        
        v_inv = 0
        delta_sum = 0
        
        for i, (outcome, other_rating, other_rd) in enumerate(zip(outcomes, other_ratings, other_rds)):
            # Convert other rating to Glicko-2 scale
            other_rating = (other_rating - 1500) / self._scale_factor
            other_rd = other_rd / self._scale_factor
            
            expected_score = self.E(rating, other_rating, other_rd)
            g_rd = self.g(other_rd)
            v_inv += (g_rd**2) * expected_score * (1 - expected_score)
            delta_sum += g_rd * (outcome - expected_score)
            
        if v_inv > 0:
            v = 1 / v_inv
            delta = delta_sum * v
            
            # Update volatility
            new_vol = self._volatility(rating, rd, vol, delta, v)
            
            # Update rating deviation
            new_rd = np.sqrt(rd**2 + new_vol**2)
            
            # Pre-rating period value
            new_rd = 1 / np.sqrt(1/new_rd**2 + 1/v)
            
            # Update rating
            new_rating = rating + new_rd**2 * delta_sum
            
            # Convert back to Glicko-1 scale
            new_rating = new_rating * self._scale_factor + 1500
            new_rd = new_rd * self._scale_factor
            
            return new_rating, new_rd, new_vol
        else:
            # If there are no games, increase the RD
            new_rd = np.sqrt(rd**2 + vol**2)
            
            # Convert back to Glicko-1 scale
            new_rd = new_rd * self._scale_factor
            
            return rating * self._scale_factor + 1500, new_rd, vol


def load_nectar_dataset(dataset_name="berkeley-nest/Nectar"):
    """
    Load the Nectar dataset from Hugging Face.
    
    Parameters:
    dataset_name (str): The dataset name on Hugging Face.
    
    Returns:
    Dataset: The loaded dataset.
    """
    try:
        dataset = load_dataset(dataset_name)
        print(f"Successfully loaded the {dataset_name} dataset")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def explore_dataset_structure(dataset, num_examples=3):
    """
    Explore the structure of the dataset to understand its format.
    
    Parameters:
    dataset: The dataset to explore.
    num_examples (int): Number of examples to examine.
    
    Returns:
    dict: Information about the dataset structure.
    """
    print("Exploring dataset structure...")
    structure_info = {}
    
    # Dataset splits
    structure_info["splits"] = list(dataset.keys())
    print(f"Dataset splits: {structure_info['splits']}")
    
    # Sample size
    train_size = len(dataset["train"])
    structure_info["train_size"] = train_size
    print(f"Training set size: {train_size} examples")
    
    # Examine sample examples
    print("\nExamining example structure:")
    sample_examples = []
    
    for i in range(min(num_examples, train_size)):
        example = dataset["train"][i]
        example_info = {"keys": list(example.keys())}
        
        print(f"\nExample {i} keys: {example_info['keys']}")
        
        # Check for trajectory/preference data
        for key in example.keys():
            value = example[key]
            value_type = type(value).__name__
            print(f"  Key: {key}, Type: {value_type}")
            
            if key == "trajectories" and isinstance(value, list):
                print(f"    Found {len(value)} trajectories")
                if len(value) > 0:
                    traj = value[0]
                    if isinstance(traj, dict):
                        print(f"    Trajectory keys: {list(traj.keys())}")
                        
                        # Check for states and actions
                        for traj_key in ["states", "observations", "actions"]:
                            if traj_key in traj:
                                traj_data = traj[traj_key]
                                print(f"    Found {traj_key} with {len(traj_data) if isinstance(traj_data, list) else 'scalar'} values")
            
            elif key == "trajectory" and isinstance(value, list):
                print(f"    Found {len(value)} trajectories")
                if len(value) > 0:
                    traj = value[0]
                    if isinstance(traj, dict):
                        print(f"    Trajectory keys: {list(traj.keys())}")
                        
                        # Check for states and actions
                        for traj_key in ["states", "observations", "actions"]:
                            if traj_key in traj:
                                traj_data = traj[traj_key]
                                print(f"    Found {traj_key} with {len(traj_data) if isinstance(traj_data, list) else 'scalar'} values")
            
            elif key in ["human_preference", "preference", "choice", "label"]:
                print(f"    Preference value: {value}")
        
        sample_examples.append(example_info)
    
    structure_info["sample_examples"] = sample_examples
    return structure_info


def extract_state_action_pairs(dataset, max_examples=None):
    """
    Extract state-action pairs and outcomes from the dataset.
    
    Parameters:
    dataset: The Nectar dataset.
    max_examples (int, optional): Maximum number of examples to use for training.
                                 If None, use all examples. Default is None.
    
    Returns:
    dict: Dictionary of state-action pairs with their ratings and outcomes.
    """
    state_action_pairs = defaultdict(lambda: {
        'ratings': 1500, 
        'rd': 350, 
        'vol': 0.06, 
        'outcomes': [], 
        'other_ratings': [], 
        'other_rds': [],
        'preferred_count': 0,
        'total_count': 0
    })
    
    # Get the train dataset
    train_data = dataset["train"]
    
    # Limit the number of examples if specified
    if max_examples is not None and max_examples > 0 and max_examples < len(train_data):
        indices = random.sample(range(len(train_data)), max_examples)
        train_subset = [train_data[i] for i in indices]
        print(f"Using {max_examples} randomly selected examples out of {len(train_data)} total examples")
    else:
        train_subset = train_data
        print(f"Using all {len(train_data)} available examples")
    
    # Try to determine the dataset structure
    first_example = train_subset[0] if train_subset else None
    if not first_example:
        print("Error: No examples found in the dataset")
        return state_action_pairs
    
    # Identify key field names
    trajectory_key = "trajectory" if "trajectory" in first_example else "trajectories"
    preference_key = None
    for key in ["human_preference", "preference", "choice", "label"]:
        if key in first_example:
            preference_key = key
            break
    
    if trajectory_key not in first_example:
        print(f"Error: Could not find trajectory data in the dataset")
        print(f"Available keys: {list(first_example.keys())}")
        return state_action_pairs
    
    print(f"Using trajectory_key='{trajectory_key}' and preference_key='{preference_key}'")
    
    # Process each example
    for item_idx, item in enumerate(tqdm(train_subset, desc="Processing examples")):
        # Extract trajectories and preference
        trajectories = item[trajectory_key]
        preference = item.get(preference_key, None) if preference_key else None
        
        if not isinstance(trajectories, list) or not trajectories:
            continue
        
        if preference is not None and not isinstance(preference, (int, float)):
            continue
        
        # Process each trajectory
        for traj_idx, traj in enumerate(trajectories):
            if not isinstance(traj, dict):
                continue
            
            # Extract states and actions
            states = None
            actions = None
            
            for state_key in ["states", "observations", "obs", "state"]:
                if state_key in traj:
                    states = traj[state_key]
                    break
            
            for action_key in ["actions", "action", "acts"]:
                if action_key in traj:
                    actions = traj[action_key]
                    break
            
            if states is None or actions is None:
                continue
            
            # Ensure states and actions are lists or arrays
            if not isinstance(states, (list, np.ndarray)) or not isinstance(actions, (list, np.ndarray)):
                continue
            
            # Create state-action pairs
            for i in range(min(len(states), len(actions))):
                # Create state and action representations
                # Convert numpy arrays to regular Python lists for consistent string representation
                state_val = states[i]
                action_val = actions[i]
                
                if isinstance(state_val, np.ndarray):
                    state_val = state_val.tolist()
                if isinstance(action_val, np.ndarray):
                    action_val = action_val.tolist()
                
                state_repr = str(state_val)
                action_repr = str(action_val)
                state_action_key = f"{state_repr}_{action_repr}"
                
                # Determine if this state-action pair was in the preferred trajectory
                is_preferred = False
                if preference is not None:
                    is_preferred = (traj_idx == preference)
                    outcome = 1.0 if is_preferred else 0.0
                else:
                    # If no preference data, use neutral outcome
                    outcome = 0.5
                
                # Update the state-action pair information
                state_action_pairs[state_action_key]['outcomes'].append(outcome)
                state_action_pairs[state_action_key]['other_ratings'].append(1500)
                state_action_pairs[state_action_key]['other_rds'].append(350)
                
                # Track preference statistics
                state_action_pairs[state_action_key]['total_count'] += 1
                if is_preferred:
                    state_action_pairs[state_action_key]['preferred_count'] += 1
    
    # Calculate preference ratio for each state-action pair
    for key, data in state_action_pairs.items():
        if data['total_count'] > 0:
            data['preference_ratio'] = data['preferred_count'] / data['total_count']
        else:
            data['preference_ratio'] = 0.0
    
    print(f"Extracted {len(state_action_pairs)} unique state-action pairs")
    return state_action_pairs


def build_reward_function(state_action_pairs, rd_scale=1.0, glicko_system=None):
    """
    Build a reward function from state-action pairs using Glicko-2 ratings.
    
    Parameters:
    state_action_pairs (dict): Dictionary of state-action pairs with their ratings and outcomes.
    rd_scale (float): Scaling factor for the rating deviation in the reward function.
                     Higher values penalize uncertainty more. Default is 1.0.
    glicko_system (Glicko2, optional): Glicko-2 system to use. If None, creates a new one.
    
    Returns:
    tuple: (reward_function, updated_state_action_pairs)
    """
    if glicko_system is None:
        glicko_system = Glicko2()
    
    reward_function = {}
    
    print(f"Building reward function with RD scaling factor: {rd_scale}")
    
    # Build the reward function
    for state_action_key, data in tqdm(state_action_pairs.items(), desc="Calculating rewards"):
        if data['outcomes']:  # Only update if there are outcomes
            new_rating, new_rd, new_vol = glicko_system.update_player(
                data['ratings'], 
                data['rd'], 
                data['vol'],
                data['outcomes'],
                data['other_ratings'],
                data['other_rds']
            )
            
            # Update the state-action pair data
            state_action_pairs[state_action_key]['ratings'] = new_rating
            state_action_pairs[state_action_key]['rd'] = new_rd
            state_action_pairs[state_action_key]['vol'] = new_vol
            
            # Define the reward as Glicko-2 score minus scaled rating deviation
            reward = new_rating - (new_rd * rd_scale)
            reward_function[state_action_key] = reward
    
    return reward_function, state_action_pairs


def evaluate_reward_function(reward_function, state_action_pairs):
    """
    Evaluate the quality of the reward function.
    
    Parameters:
    reward_function (dict): The reward function mapping state-action pairs to rewards.
    state_action_pairs (dict): Dictionary of state-action pairs with their ratings and stats.
    
    Returns:
    dict: Evaluation metrics.
    """
    if not reward_function:
        return {"error": "Empty reward function"}
    
    # Create DataFrames for analysis
    df_rewards = pd.DataFrame([
        {
            'state_action': key,
            'reward': value,
            'rating': state_action_pairs[key]['ratings'],
            'rd': state_action_pairs[key]['rd'],
            'volatility': state_action_pairs[key]['vol'],
            'preference_ratio': state_action_pairs[key].get('preference_ratio', 0),
            'total_count': state_action_pairs[key].get('total_count', 0)
        }
        for key, value in reward_function.items()
    ])
    
    # Calculate various metrics
    metrics = {
        'num_state_actions': len(reward_function),
        'mean_reward': df_rewards['reward'].mean(),
        'std_reward': df_rewards['reward'].std(),
        'min_reward': df_rewards['reward'].min(),
        'max_reward': df_rewards['reward'].max(),
        'mean_rating': df_rewards['rating'].mean(),
        'mean_rd': df_rewards['rd'].mean(),
        'correlation_rating_preference': df_rewards[['rating', 'preference_ratio']].corr().iloc[0, 1],
        'correlation_reward_preference': df_rewards[['reward', 'preference_ratio']].corr().iloc[0, 1]
    }
    
    print("\nReward Function Evaluation:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    return metrics, df_rewards


def visualize_reward_function(df_rewards, rd_scale, output_prefix=""):
    """
    Create visualizations of the reward function.
    
    Parameters:
    df_rewards (DataFrame): DataFrame with reward function data.
    rd_scale (float): The RD scaling factor used.
    output_prefix (str): Prefix for output filenames.
    """
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create visualizations
    
    # 1. Distribution of rewards
    plt.figure(figsize=(12, 6))
    sns.histplot(df_rewards['reward'], kde=True)
    plt.title(f'Distribution of Rewards (Rating - {rd_scale} × RD)')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    if output_prefix:
        plt.savefig(f"{output_prefix}_reward_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Rating vs. Rating Deviation
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='rating', y='rd', hue='preference_ratio', size='total_count',
                   sizes=(20, 200), data=df_rewards, alpha=0.7)
    plt.title('Rating vs. Rating Deviation')
    plt.xlabel('Glicko-2 Rating')
    plt.ylabel('Rating Deviation (RD)')
    plt.grid(True, alpha=0.3)
    if output_prefix:
        plt.savefig(f"{output_prefix}_rating_vs_rd.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Reward vs. Preference Ratio
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='reward', y='preference_ratio', hue='total_count', 
                   size='total_count', sizes=(20, 200), data=df_rewards, alpha=0.7)
    plt.title('Reward vs. Preference Ratio')
    plt.xlabel('Reward')
    plt.ylabel('Preference Ratio (Higher = More Preferred)')
    plt.grid(True, alpha=0.3)
    if output_prefix:
        plt.savefig(f"{output_prefix}_reward_vs_preference.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Top and Bottom State-Actions by Reward
    top_actions = df_rewards.nlargest(10, 'reward')
    bottom_actions = df_rewards.nsmallest(10, 'reward')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top actions
    sns.barplot(x='reward', y=top_actions.index.astype(str), data=top_actions, ax=axes[0])
    axes[0].set_title('Top 10 State-Actions by Reward')
    axes[0].set_xlabel('Reward')
    axes[0].set_ylabel('State-Action Index')
    
    # Bottom actions
    sns.barplot(x='reward', y=bottom_actions.index.astype(str), data=bottom_actions, ax=axes[1])
    axes[1].set_title('Bottom 10 State-Actions by Reward')
    axes[1].set_xlabel('Reward')
    axes[1].set_ylabel('State-Action Index')
    
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_top_bottom_rewards.png", dpi=300, bbox_inches='tight')
    plt.show()


def hyperparameter_experiment(dataset, rd_scale_values=[0.5, 1.0, 2.0, 5.0], 
                             max_examples_values=[None, 1000, 5000]):
    """
    Run experiments with different hyperparameter values.
    
    Parameters:
    dataset: The Nectar dataset.
    rd_scale_values (list): List of RD scaling factors to try.
    max_examples_values (list): List of values for max_examples to try.
    
    Returns:
    dict: Results for each hyperparameter combination.
    """
    results = {}
    experiment_data = []
    
    for max_ex in max_examples_values:
        # Extract state-action pairs once for each max_examples value
        print(f"\n=== Extracting data for max_examples={max_ex} ===")
        sa_pairs = extract_state_action_pairs(dataset, max_examples=max_ex)
        
        for rd_scale in rd_scale_values:
            print(f"\n=== Experiment with max_examples={max_ex}, rd_scale={rd_scale} ===")
            
            # Build reward function
            reward_func, updated_pairs = build_reward_function(
                sa_pairs.copy(), rd_scale=rd_scale)
            
            # Evaluate the reward function
            metrics, df = evaluate_reward_function(reward_func, updated_pairs)
            
            # Store results
            experiment_id = f"max_ex={max_ex},rd_scale={rd_scale}"
            results[experiment_id] = {
                'max_examples': max_ex,
                'rd_scale': rd_scale,
                'metrics': metrics
            }
            
            # Add to experiment data for DataFrame
            experiment_data.append({
                'max_examples': str(max_ex),
                'rd_scale': rd_scale,
                'num_state_actions': metrics['num_state_actions'],
                'mean_reward': metrics['mean_reward'],
                'correlation_reward_preference': metrics['correlation_reward_preference']
            })
    
    # Create DataFrame for visualization
    df_experiments = pd.DataFrame(experiment_data)
    
    # Visualize experiment results
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='rd_scale', y='correlation_reward_preference', hue='max_examples', 
                 marker='o', data=df_experiments)
    plt.title('Effect of RD Scale on Correlation between Reward and Preference')
    plt.xlabel('RD Scale Factor')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True, alpha=0.3)
    plt.savefig("hyperparameter_experiment_correlation.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, df_experiments


def get_reward(state, action, reward_func, default_value=1500):
    """
    Get the reward for a given state-action pair.
    
    Parameters:
    state: The current state.
    action: The action to evaluate.
    reward_func (dict): The reward function to use.
    default_value (float, optional): Default reward value to return if state-action pair not found.
    
    Returns:
    float: The reward value for the state-action pair.
    """
    # Convert numpy arrays to lists for consistent representation
    if isinstance(state, np.ndarray):
        state = state.tolist()
    if isinstance(action, np.ndarray):
        action = action.tolist()
    
    state_repr = str(state)
    action_repr = str(action)
    state_action_key = f"{state_repr}_{action_repr}"
    
    if state_action_key in reward_func:
        return reward_func[state_action_key]
    else:
        return default_value


def save_reward_function(reward_function, state_action_pairs, hyperparameters, filename):
    """
    Save the reward function and related data to a file.
    
    Parameters:
    reward_function (dict): The reward function.
    state_action_pairs (dict): The state-action pairs data.
    hyperparameters (dict): Hyperparameters used.
    filename (str): Output filename.
    """
    data = {
        'reward_function': reward_function,
        'state_action_pairs': state_action_pairs,
        'hyperparameters': hyperparameters,
        'metadata': {
            'num_state_actions': len(reward_function),
            'creation_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Reward function saved to '{filename}'")


def load_reward_function(filename):
    """
    Load a saved reward function.
    
    Parameters:
    filename (str): Input filename.
    
    Returns:
    dict: The loaded data.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded reward function with {len(data['reward_function'])} state-action pairs")
    print(f"Hyperparameters: {data['hyperparameters']}")
    
    return data


def main():
    """Main function to run the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a Glicko-2 based reward function on the Nectar dataset')
    parser.add_argument('--rd_scale', type=float, default=1.0, 
                        help='Scaling factor for rating deviation in the reward function')
    parser.add_argument('--max_examples', type=int, default=None, 
                        help='Maximum number of examples to use (None to use all)')
    parser.add_argument('--output', type=str, default='glicko2_reward_function.pkl',
                        help='Output filename for the reward function')
    parser.add_argument('--explore', action='store_true',
                        help='Explore the dataset structure before processing')
    parser.add_argument('--experiment', action='store_true',
                        help='Run hyperparameter experiments')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of the reward function')
    
    args = parser.parse_args()
    
    # Load the dataset
    print("Loading Nectar dataset...")
    dataset = load_nectar_dataset()
    
    # Explore dataset structure if requested
    if args.explore:
        structure_info = explore_dataset_structure(dataset)
    
    # Run hyperparameter experiments if requested
    if args.experiment:
        print("\n=== Running Hyperparameter Experiments ===")
        experiment_results, df_experiments = hyperparameter_experiment(
            dataset, 
            rd_scale_values=[0.5, 1.0, 2.0, 5.0],
            max_examples_values=[None, 1000, 5000]
        )
        
        print("\nExperiment Results:")
        print(df_experiments)
        
        # Save experiment results
        df_experiments.to_csv("glicko2_experiments.csv", index=False)
        with open("glicko2_experiment_results.pkl", "wb") as f:
            pickle.dump(experiment_results, f)
        
        print("Experiment results saved to 'glicko2_experiments.csv' and 'glicko2_experiment_results.pkl'")
        return
    
    # Train the