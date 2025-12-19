import gymnasium as gym
import torch
import numpy as np
import random
from tqdm import tqdm
from collections import deque

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
ENV_ID = "FrozenLake-v1"
IS_SLIPPERY = False
MAX_STEPS_PER_EPISODE = 100
GAMMA = 0.99
ALPHA_PSI = 0.15
ALPHA_W = 0.05
EPSILON_START = 1.0
EPSILON_END = 0.01
DECAY_EPISODES = 2000 # Epsilon decay duration
MAX_TRAIN_EPISODES = 5000 # Hard limit if not converged

# Custom Map (4x4)
CUSTOM_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFF",
]

# Goals
TASK_1_GOAL = 15   # (Row 4, Col 4)
#TASK_2_GOAL = 10  # (Row 3, Col 3) - Bottom Right
goals=[1,2,3,4,6,8,9,10,13,14,15]

class TaskRewardWrapper(gym.Wrapper):
    """
    Non-terminating goal wrapper.
    Reward is 1.0 if agent is in goal_state, 0.0 otherwise.
    Episode only ends on 'Hole' (base env termination) or Timeout.
    """
    def __init__(self, env, goal_state: int):
        super().__init__(env)
        self.goal_state = goal_state

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        s = int(obs) if not isinstance(obs, (tuple, list)) else int(obs[0])
        
        # Reward logic: 1.0 if at goal, else 0.0
        reward = 1.0 if s == self.goal_state else 0.0
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def get_theoretical_max_return(goal_state, desc):
    """
    Calculates the theoretical max return for specific goal.
    Return = (MaxSteps - DistanceToGoal).
    Assumes non-slippery and agent stays at goal receiving +1 per step.
    """
    n_rows = len(desc)
    n_cols = len(desc[0])
    
    # BFS to find shortest path from Start (0) to Goal
    queue = deque([(0, 0)]) # state, steps
    visited = {0}
    shortest_steps = float('inf')
    
    while queue:
        s, steps = queue.popleft()
        if s == goal_state:
            shortest_steps = steps
            break
        
        row, col = s // n_cols, s % n_cols
        
        # Directions: Left, Down, Right, Up
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                char = desc[nr][nc]
                if char != 'H': # Valid move
                    ns = nr * n_cols + nc
                    if ns not in visited:
                        visited.add(ns)
                        queue.append((ns, steps + 1))

    if shortest_steps == float('inf'):
        return 0.0
    
    # Calculation:
    # If it takes L steps to reach, we have (100 - L) steps remaining.
    # We get 1.0 reward on the step we arrive? 
    # Logic: s_prev -> action -> s_goal (Reward 1.0).
    # So we get reward for the transition INTO the goal and every stay.
    # Total Reward = 100 - shortest_steps
    return float(MAX_STEPS_PER_EPISODE - shortest_steps)

# ==========================================
# 2. TRAINING FUNCTION
# ==========================================
def train_agent(
    run_name: str,
    goal_state: int,
    initial_psi: torch.Tensor = None,
    decorrelation_x: float = 0.0,
    check_convergence: bool = False
):
    """
    Args:
        run_name: Label for logging.
        goal_state: Target state integer.
        initial_psi: Tensor to initialize Psi table. If None, use Zeros.
        decorrelation_x: Strength of decorrelation (0.0 = Standard SF).
        check_convergence: If True, stops when optimal return is hit 10x continuously.
    Returns:
        trained_psi, episodes_taken
    """
    
    # Setup Env
    base_env = gym.make(ENV_ID, desc=CUSTOM_MAP, is_slippery=IS_SLIPPERY, render_mode=None)
    env = TaskRewardWrapper(base_env, goal_state=goal_state)
    
    S = env.observation_space.n
    A = env.action_space.n
    d = S * A 
    N = S * A

    # Initialize Tables
    if initial_psi is not None:
        psi = initial_psi.clone()
    else:
        psi = torch.zeros((S, A, d), dtype=torch.float32)
        
    w = torch.zeros((d,), dtype=torch.float32)

    # Determine Convergence Threshold
    max_theoretical_return = get_theoretical_max_return(goal_state, CUSTOM_MAP)
    # We allow a tiny margin of error or exact match
    optimality_threshold = max_theoretical_return 
    
    print(f"\n--- Starting: {run_name} (Goal: {goal_state}, x={decorrelation_x}) ---")
    if check_convergence:
        print(f"Target: {optimality_threshold} return for 10 consecutive eps.")

    # Tracking
    return_history = deque(maxlen=10)
    converged_at = -1
    
    with tqdm(total=MAX_TRAIN_EPISODES, desc=run_name, leave=True) as pbar:
        for episode in range(1, MAX_TRAIN_EPISODES + 1):
            
            # Epsilon Schedule
            if episode <= DECAY_EPISODES:
                frac = (episode - 1) / max(1, DECAY_EPISODES - 1)
                epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)
            else:
                epsilon = EPSILON_END

            obs, _ = env.reset()
            s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)
            ep_return = 0.0

            for step in range(MAX_STEPS_PER_EPISODE):
                # 1. Action Selection
                q_vals = torch.matmul(psi[s], w)
                if random.random() < epsilon:
                    a = random.randrange(A)
                else:
                    a = int(torch.argmax(q_vals))

                # 2. Step
                step_result = env.step(a)
                if len(step_result) == 5:
                    s_dash_raw, r, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    s_dash_raw, r, done, _ = step_result
                
                s_dash = int(s_dash_raw) if not isinstance(s_dash_raw, (tuple, list)) else int(s_dash_raw[0])

                # 3. Next Action (Greedy) for Bootstrap
                q_vals_next = torch.matmul(psi[s_dash], w)
                a_dash = int(torch.argmax(q_vals_next))

                # 4. Construct Feature phi
                idx = s * A + a
                phi = torch.zeros(d)
                phi[idx] = 1.0

                # 5. TD Errors
                w_idx = idx # w is shape (d,)
                w[w_idx] += ALPHA_W * (r - w[w_idx])
                
                psi_pred = psi[s, a]
                psi_target = phi + GAMMA * psi[s_dash, a_dash]
                psi_error = psi_target - psi_pred

                # 6. Decorrelation Gradient (if x > 0)
                grad_row = torch.zeros(d)
                if decorrelation_x > 0:
                    Psi = psi.reshape(N, d)
                    Psi_mean = Psi.mean(dim=0, keepdim=True)
                    C = Psi - Psi_mean
                    Cov = (C.T @ C) / N
                    I = torch.eye(d)
                    GradPsi = (4.0 / N) * (C @ (Cov - I))
                    grad_row = GradPsi[idx]

                # 7. Update Psi
                psi[s, a] += ALPHA_PSI * (psi_error - decorrelation_x * grad_row)

                ep_return += r
                s = s_dash
                
                if done:
                    break
            
            # --- End of Episode Checks ---
            return_history.append(ep_return)
            
            # Check Convergence Condition
            if check_convergence:
                if len(return_history) == 10:
                    # Check if ALL last 10 episodes achieved max theoretical return
                    if all(ret >= optimality_threshold for ret in return_history):
                        converged_at = episode
                        pbar.set_description(f"{run_name} [CONVERGED @ {episode}]")
                        break
            
            pbar.update(1)
            pbar.set_postfix({'ret': f"{ep_return:.1f}", 'eps': f"{epsilon:.2f}"})

    env.close()
    
    if converged_at == -1:
        print(f"Did not converge within {MAX_TRAIN_EPISODES} episodes.")
        return psi, MAX_TRAIN_EPISODES
    else:
        print(f"Converged in {converged_at} episodes.")
        return psi, converged_at

# ==========================================
# 3. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    print("=================================================")
    print("PHASE 1: PRE-TRAINING ON TASK 1 (Goal 9)")
    print("=================================================")
    
    # 1. Train with Decorrelation (x=20) -> psi_bar
    psi_bar, _ = train_agent(
        run_name="Task1_Decorrelated", 
        goal_state=TASK_1_GOAL, 
        initial_psi=None, 
        decorrelation_x=20.0, 
        check_convergence=True
    )
    
    # 2. Train with Standard SF (x=0) -> psi_optimal
    psi_optimal, _ = train_agent(
        run_name="Task1_Standard", 
        goal_state=TASK_1_GOAL, 
        initial_psi=None, 
        decorrelation_x=0.0, 
        check_convergence=True
    )

    print("\n=================================================")
    print(f"PHASE 2: TRANSFER TO TASK 2 (Goal {goals})")
    print("comparing Time-to-Optimality (10 continuous max returns)")
    print("=================================================")

    # Define common args for Task 2 training
    # Note: We use x=0 for Task 2 to test the initialization quality purely,
    # unless you want to continue regularizing. Usually transfer is tested with standard training.
    
    # 3. Init from psi_bar
    for TASK_2_GOAL in goals:
        _, epochs_bar = train_agent(
            run_name="Task2_Init_PsiBar",
            goal_state=TASK_2_GOAL,
            initial_psi=psi_bar,
            decorrelation_x=0.0, 
            check_convergence=True
        )
    
        # 4. Init from psi_optimal
        _, epochs_opt = train_agent(
            run_name="Task2_Init_PsiOpt",
            goal_state=TASK_2_GOAL,
            initial_psi=psi_optimal,
            decorrelation_x=0.0,
            check_convergence=True
        )
        
        # 5. Init from Zeros
        _, epochs_zero = train_agent(
            run_name="Task2_Init_Zeros",
            goal_state=TASK_2_GOAL,
            initial_psi=None,
            decorrelation_x=0.0,
            check_convergence=True
        )

        print("\n\n=================================================")
        print(f"RESULTS: EPISODES TO OPTIMALITY ON TASK 2 goal: {TASK_2_GOAL}")
        print("=================================================")
        print(f"{'Initialization Strategy':<30} | {'Episodes to Converge':<20}")
        print("-" * 55)
        print(f"{'Decorrelated (psi_bar)':<30} | {epochs_bar:<20}")
        print(f"{'Standard (psi_optimal)':<30} | {epochs_opt:<20}")
        print(f"{'Scratch (Zeros)':<30} | {epochs_zero:<20}")
        print("=================================================")
