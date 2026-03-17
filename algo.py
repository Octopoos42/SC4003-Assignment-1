import numpy as np
from const import actions, transition_probs, right_moves

def get_next_state(state, action, valid_non_wall_states, maze):
    """
    Compute the next state after taking an action in the maze.

    Parameters
    ----------
    state : tuple[int, int]
        Current grid coordinates as (row, col).
    action : str
        One of the action symbols in `actions` (e.g., "U", "D", "L", "R").
    valid_non_wall_states : set[tuple[int, int]]
        Set of coordinates that are valid (not walls).
    maze : np.ndarray or similar
        Grid representation used for indexing rewards or cell types.

    Returns
    -------
    tuple[int, int]
        The new state after applying the action. If the intended move would
        hit a wall or an invalid cell, returns the original state.
    """

    r, c = state

    if action == "U" and (r - 1, c) in valid_non_wall_states:
        return (r - 1, c)
    
    if action == "D" and (r + 1, c) in valid_non_wall_states:
        return (r + 1, c)
    
    if action == "L" and (r, c - 1) in valid_non_wall_states:
        return (r, c - 1)
    
    if action == "R" and (r, c + 1) in valid_non_wall_states:
        return (r, c + 1)
    
    return state

def get_max_reward_abs(rewards):
    """
    Return the maximum absolute reward value from the rewards mapping.

    Parameters
    ----------
    rewards : dict
        Mapping from maze cell identifier to numeric reward.

    Returns
    -------
    float
        The largest absolute reward value among all entries in `rewards`.
    """

    return max(abs(r) for r in rewards.values())

def value_iteration(rewards, maze, valid_non_wall_states, C=0.01, gamma=0.99, max_iters=2000):
    """
    Run value iteration for the MDP defined by the maze and rewards.

    Parameters
    ----------
    rewards : dict
        Mapping from maze cell identifier to numeric reward.
    maze : np.ndarray
        2D array representing the maze; used to index into `rewards`.
    valid_non_wall_states : iterable[tuple[int, int]]
        Iterable of coordinates considered valid states (non-wall).
    C : float, optional
        Convergence constant used to compute epsilon (default 0.01).
    gamma : float, optional
        Discount factor for future rewards (default 0.99).
    max_iters : int, optional
        Maximum number of iterations to run (default 2000).

    Returns
    -------
    utilities : dict
        Mapping from state to computed utility values.
    policy : dict
        Greedy policy mapping from state to best action.
    utility_history : dict
        Per-state list of utility values across iterations.
    policy_history : list
        List of policy snapshots (dict copies) per iteration.
    iterations : int
        Number of iterations performed (or max_iters if not converged).

    Notes
    -----
    - Uses `transition_probs` and `right_moves` to compute expected utilities
      under stochastic transitions (intended + right-angle slips).
    - Convergence threshold is computed from C and the maximum absolute reward.
    - Policy is updated greedily each iteration.
    - Time complexity per iteration is O(|S| * |A|) where |S| is number of states
      and |A| is number of actions.
    """

    epsilon = C * get_max_reward_abs(rewards)
    threshold = epsilon * (1 - gamma) / gamma
    utilities = {s: 0 for s in valid_non_wall_states}
    policy = {s: np.random.choice(actions) for s in valid_non_wall_states}
    utility_history = {s: [] for s in valid_non_wall_states}
    policy_history = []
    
    for i in range(max_iters):
        new_utilities = utilities.copy()
        delta = 0
        
        for state in valid_non_wall_states:
            max_utility, best_action = float("-inf"), None
            
            for action in actions:
                expected_utility = transition_probs["intended"] * utilities[get_next_state(state, action, valid_non_wall_states, maze)]
                
                for ra in right_moves[action]:
                    expected_utility += transition_probs["right_angle"] * utilities[get_next_state(state, ra, valid_non_wall_states, maze)]
                
                if expected_utility > max_utility:
                    max_utility, best_action = expected_utility, action
            
            new_utilities[state] = rewards[maze[state[0], state[1]]] + gamma * max_utility
            policy[state] = best_action
            delta = max(delta, abs(new_utilities[state] - utilities[state]))
            utility_history[state].append(new_utilities[state])
        
        policy_history.append(policy.copy())
        
        if delta < threshold:
            return utilities, policy, utility_history, policy_history, i
        
        utilities = new_utilities
    
    return utilities, policy, utility_history, policy_history, max_iters

def bellman_equation_pi(state, action, utilities, rewards, maze, valid_non_wall_states, gamma):
    """
    Evaluate the Bellman update for a given state and action using current utilities.

    Parameters
    ----------
    state : tuple[int, int]
        The state for which to evaluate the Bellman expression.
    action : str
        Action to evaluate (one of the action symbols).
    utilities : dict
        Current utility estimates for all states.
    rewards : dict
        Mapping from maze cell identifier to numeric reward.
    maze : np.ndarray
        2D array representing the maze; used to index into `rewards`.
    valid_non_wall_states : set[tuple[int, int]]
        Set of valid states (used indirectly by get_next_state).
    gamma : float
        Discount factor.

    Returns
    -------
    float
        The immediate reward for `state` plus discounted expected utility
        after taking `action` (accounts for stochastic transitions).
    """

    reward = rewards[maze[state[0], state[1]]]
    expected_util = transition_probs["intended"] * utilities[get_next_state(state, action, valid_non_wall_states, maze)]
    
    for ra in right_moves[action]:
        expected_util += transition_probs["right_angle"] * utilities[get_next_state(state, ra, valid_non_wall_states, maze)]
    
    return reward + gamma * expected_util

def evaluate_policy(policy, rewards, maze, valid_non_wall_states, gamma=0.99, C=0.01):
    """
    Evaluate a given policy by iteratively solving the Bellman equations.

    Parameters
    ----------
    policy : dict
        Mapping from state to action (the policy to evaluate).
    rewards : dict
        Mapping from maze cell identifier to numeric reward.
    maze : np.ndarray
        2D array representing the maze; used to index into `rewards`.
    valid_non_wall_states : iterable[tuple[int, int]]
        Iterable of coordinates considered valid states (non-wall).
    gamma : float, optional
        Discount factor (default 0.99).
    C : float, optional
        Convergence constant used to compute epsilon (default 0.01).

    Returns
    -------
    utilities : dict
        Utility estimates for each state under the provided policy.

    Notes
    -----
    - Iterates until the maximum change across states is below the threshold
      derived from C and the maximum absolute reward.
    - Uses `bellman_equation_pi` to compute each state's update.
    - Time complexity per evaluation iteration is O(|S|).
    """

    epsilon = C * get_max_reward_abs(rewards)
    threshold = epsilon * (1 - gamma) / gamma
    utilities = {s: 0 for s in valid_non_wall_states}
    
    while True:
        new_utilities = utilities.copy()
        delta = 0
        
        for state in valid_non_wall_states:
            action = policy[state]
            new_utilities[state] = bellman_equation_pi(state, action, utilities, rewards, maze, valid_non_wall_states, gamma)
            delta = max(delta, abs(new_utilities[state] - utilities[state]))
        utilities = new_utilities
        
        if delta < threshold:
            break
    
    return utilities

def policy_iteration(rewards, maze, valid_non_wall_states, gamma=0.99, max_iters=500, C=0.01):
    """
    Perform policy iteration to find an optimal policy and its utilities.

    Parameters
    ----------
    rewards : dict
        Mapping from maze cell identifier to numeric reward.
    maze : np.ndarray
        2D array representing the maze; used to index into `rewards`.
    valid_non_wall_states : iterable[tuple[int, int]]
        Iterable of coordinates considered valid states (non-wall).
    gamma : float, optional
        Discount factor (default 0.99).
    max_iters : int, optional
        Maximum number of policy-improvement iterations (default 500).
    C : float, optional
        Convergence constant passed to policy evaluation (default 0.01).

    Returns
    -------
    utilities : dict
        Utility estimates for the final policy.
    policy : dict
        Final (stable) policy mapping from state to action.
    utility_history : dict
        Per-state list of utility values across policy-iteration steps.
    policy_history : list
        List of policy snapshots (dict copies) per iteration.
    iterations : int
        Number of policy-iteration steps performed.

    Notes
    -----
    - Uses `evaluate_policy` for the policy-evaluation step.
    - Improves the policy greedily using `bellman_equation_pi`.
    - Stops early if the policy is stable (no changes).
    - Complexity depends on evaluation cost and number of improvements; often
      converges in fewer iterations than value iteration for many problems.
    """

    policy = {s: np.random.choice(actions) for s in valid_non_wall_states}
    utility_history = {s: [] for s in valid_non_wall_states}
    policy_history = []
    
    for i in range(max_iters):
        utilities = evaluate_policy(policy, rewards, maze, valid_non_wall_states, gamma, C)
        
        for state in valid_non_wall_states:
            utility_history[state].append(utilities[state])
        
        new_policy = {}
        policy_stable = True
        
        for state in valid_non_wall_states:
            max_utility, best_action = float("-inf"), None
            
            for action in actions:
                utility = bellman_equation_pi(state, action, utilities, rewards, maze, valid_non_wall_states, gamma)
                
                if utility > max_utility:
                    max_utility = utility
                    best_action = action
            
            new_policy[state] = best_action
            
            if new_policy[state] != policy[state]:
                policy_stable = False
        
        policy_history.append(new_policy.copy())
        
        if policy_stable:
            return utilities, new_policy, utility_history, policy_history, i + 1
        
        policy = new_policy
    
    return utilities, policy, utility_history, policy_history, max_iters

def run_mdp(rewards, maze, valid_non_wall_states, gamma=0.99, max_iters=2000, C=0.01):
    """
    Execute both value iteration and policy iteration and return their outputs.

    Parameters
    ----------
    rewards : dict
        Mapping from maze cell identifier to numeric reward.
    maze : np.ndarray
        2D array representing the maze; used to index into `rewards`.
    valid_non_wall_states : iterable[tuple[int, int]]
        Iterable of coordinates considered valid states (non-wall).
    gamma : float, optional
        Discount factor (default 0.99).
    max_iters : int, optional
        Maximum iterations for value iteration and policy iteration (default 2000).
    C : float, optional
        Convergence constant (default 0.01).

    Returns
    -------
    utilities_vi, policy_vi, uh_vi, ph_vi, steps_vi,
    utilities_pi, policy_pi, uh_pi, ph_pi, steps_pi : tuple
        Results from value iteration followed by results from policy iteration.
        Each group contains utilities, policy, utility history, policy history,
        and the number of iterations performed.
    """

    utilities_vi, policy_vi, uh_vi, ph_vi, steps_vi = value_iteration(rewards, maze, valid_non_wall_states, C, gamma, max_iters)
    utilities_pi, policy_pi, uh_pi, ph_pi, steps_pi = policy_iteration(rewards, maze, valid_non_wall_states, gamma, max_iters, C)
    return utilities_vi, policy_vi, uh_vi, ph_vi, steps_vi, utilities_pi, policy_pi, uh_pi, ph_pi, steps_pi