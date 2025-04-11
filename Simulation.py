#!/usr/bin/env python
"""
Standalone Grid Simulation using a Trained PPO Model
(Using Environment Definition Updated to Match Final Training Model)

This script defines a custom taxi environment (ModifiedTaxiEnv) that simulates a grid
with a hospital and patients, matching the final environment used in training.
It loads the previously trained PPO model and runs simulation episodes.

Requirements:
  - gymnasium
  - stable-baselines3
  - pygame
  - numpy
  - torch

Ensure you have installed these libraries (e.g., via pip install gymnasium stable-baselines3 pygame numpy torch).
"""

import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
import time
import os
import math
from stable_baselines3 import PPO

# -------------------------------
# Global Constants
# -------------------------------
GRID_SIZE = 5
HOSPITAL_LOCATION = (0, 0)
DEBUG = 0

# Simulation Configuration
MAX_NUM_PATIENTS_FOR_SIM = 5  # Number of patients for simulation (final stage)
MAX_NUM_PATIENTS_ACROSS_STAGES = 5
DISTANCE = 3  # Maximum distance for considering second pickup

# Environment Rewards/Penalties
MOVE_COST = -1
PICKUP_REWARD_CRITICAL = 250
PICKUP_REWARD_NON_CRITICAL = 125
SECOND_PICKUP_BONUS = 150  # Added bonus for double pickup
DROP_OFF_REWARD_CRITICAL = 250
DROP_OFF_REWARD_NON_CRITICAL = 125
TASK_COMPLETED_REWARD = 100
PICKUP_FAILURE_PENALTY = -5
DROPOFF_FAILURE_PENALTY = -5
TIME_EXPIRED_PENALTY = -50

# State Dimension Calculation (includes capacity indicators)
MAX_STATE_DIM = 2 + 2 + 4 + 4 * MAX_NUM_PATIENTS_ACROSS_STAGES  # taxi + target + capacity + patients
PADDING_VALUE = -3.0

# Default simulation parameters
DEFAULT_DEADLINE_CRITICAL = 20
DEFAULT_DEADLINE_NON_CRITICAL = 40
DEFAULT_MAX_STEPS = 200

# -------------------------------
# Time Decay Functions
# -------------------------------
def calculate_time_decay(wait_time, deadline, min_factor=0.3):
    """
    Calculate time decay factor based on sigmoid function
    Returns a value between min_factor and 1.0
    """
    if deadline <= 0:
        return 1.0
        
    normalized_time = wait_time / deadline
    
    # Sigmoid decay centered at half the deadline
    midpoint = 0.5
    decay_rate = 1.0
    factor = 1.0 / (1.0 + math.exp(decay_rate * (normalized_time - midpoint)))
    
    return max(min_factor, factor)

# ----------------------------------------------------
# Custom Environment Definition (Updated to Match Final Version)
# ----------------------------------------------------
class ModifiedTaxiEnv(gymnasium.Env):
    """
    Custom Gymnasium environment for multi-patient pickup and delivery with deadlines.
    Modified to match the exact observation space of the trained model.
    """
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None,
                 num_patients=MAX_NUM_PATIENTS_FOR_SIM,
                 deadline_critical=DEFAULT_DEADLINE_CRITICAL,
                 deadline_non_critical=DEFAULT_DEADLINE_NON_CRITICAL,
                 max_steps=DEFAULT_MAX_STEPS,
                 max_total_patients=MAX_NUM_PATIENTS_ACROSS_STAGES,
                 current_overall_timesteps=0):

        super().__init__()

        # Store configuration
        self.num_patients_current_stage = num_patients
        self.max_total_patients = max_total_patients
        self.hospital = HOSPITAL_LOCATION
        self._max_episode_steps = max_steps
        self.deadline_critical = deadline_critical
        self.deadline_non_critical = deadline_non_critical
        self.grid_size = GRID_SIZE
        self.current_overall_timesteps = current_overall_timesteps
        self.dist = DISTANCE
        self.second_patient_pickups = 0

        # Define Action Space (Discrete: 0-5)
        self.action_space = spaces.Discrete(6)

        # Define Observation Space to EXACTLY match the error message bounds
        # This is crucial for loading the trained model
        self.padding_value = -3.0
        
        # From error message: high bounds are:
        # [4. 4. 4. 4. 2. 1. 1. 1. 4. 4. 1. 2. 4. 4. 1. 2. 4. 4. 1. 2. 4. 4. 1. 2. 4. 4. 1. 2.]
        # We need to recreate this exact structure
        
        low = np.array([
            0.0, 0.0,             # taxi_r, taxi_c
            -1.0, -1.0,           # target_r, target_c
            0.0, 0.0, 0.0, 0.0,   # capacity indicators
            # The remaining values are patient data plus padding
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0, -3.0, -3.0
        ], dtype=np.float32)

        # This matches EXACTLY the high bounds from the error message
        high = np.array([
            4.0, 4.0,             # taxi_r, taxi_c
            4.0, 4.0,             # target_r, target_c
            2.0, 1.0, 1.0, 1.0,   # capacity indicators
            # Patient data - exact pattern from error message
            4.0, 4.0, 1.0, 2.0, 4.0, 4.0, 1.0, 2.0,
            4.0, 4.0, 1.0, 2.0, 4.0, 4.0, 1.0, 2.0,
            4.0, 4.0, 1.0, 2.0
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, shape=(28,), dtype=np.float32)

        # Internal environment state variables
        self.taxi_row, self.taxi_col = 0, 0
        self.patients = []
        self.passengers = []
        self.current_step = 0
        self.patient_wait_times = []

        # Rendering variables
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        """
        Constructs observation matching the exact structure expected by the trained model.
        """
        # 1. Agent Location
        state = [float(self.taxi_row), float(self.taxi_col)]

        # 2. Determine and Add Current Target Location
        current_target_r, current_target_c = -1.0, -1.0  # Default: invalid/no specific target

        # Check if carrying patients and their types
        carrying_patients = [i for i in range(self.num_patients_current_stage) if self.patients[i]['status'] == 1]
        carrying_critical = any(self.patients[i]['type'] == 1 for i in carrying_patients)
        waiting_patients = [i for i in range(self.num_patients_current_stage) if self.patients[i]['status'] == 0]
        waiting_non_critical = [i for i in waiting_patients if self.patients[i]['type'] == 0]

        # Target selection logic (same as before)
        if carrying_patients:
            if carrying_critical or len(carrying_patients) >= 2:
                current_target_r, current_target_c = float(self.hospital[0]), float(self.hospital[1])
            elif len(carrying_patients) == 1 and waiting_non_critical:
                taxi_pos = (self.taxi_row, self.taxi_col)
                closest_idx = None
                closest_dist = float('inf')

                for idx in waiting_non_critical:
                    patient_pos = self.patients[idx]['pos']
                    dist = abs(taxi_pos[0] - patient_pos[0]) + abs(taxi_pos[1] - patient_pos[1])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = idx

                hospital_dist = abs(taxi_pos[0] - self.hospital[0]) + abs(taxi_pos[1] - self.hospital[1])
                if closest_idx is not None and (closest_dist <= self.dist or closest_dist < hospital_dist):
                    target_pos = self.patients[closest_idx]['pos']
                    current_target_r, current_target_c = float(target_pos[0]), float(target_pos[1])
                else:
                    current_target_r, current_target_c = float(self.hospital[0]), float(self.hospital[1])
            else:
                current_target_r, current_target_c = float(self.hospital[0]), float(self.hospital[1])
        elif waiting_patients:
            taxi_pos = (self.taxi_row, self.taxi_col)
            closest_idx = None
            closest_dist = float('inf')
            
            for idx in waiting_patients:
                patient_pos = self.patients[idx]['pos']
                dist = abs(taxi_pos[0] - patient_pos[0]) + abs(taxi_pos[1] - patient_pos[1])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = idx
            
            if closest_idx is not None:
                target_pos = self.patients[closest_idx]['pos']
                current_target_r, current_target_c = float(target_pos[0]), float(target_pos[1])
        else:
            all_delivered = all(p['status'] == 2 for p in self.patients)
            if all_delivered and (self.taxi_row, self.taxi_col) != self.hospital:
                current_target_r, current_target_c = float(self.hospital[0]), float(self.hospital[1])

        state.extend([current_target_r, current_target_c])

        # 3. Add capacity indicators
        carrying_non_critical = sum(1 for p in self.patients if p['status'] == 1 and p['type'] == 0)
        carrying_critical = sum(1 for p in self.patients if p['status'] == 1 and p['type'] == 1)
        can_pickup_non_critical = 1.0 if (carrying_critical == 0 and carrying_non_critical < 2) else 0.0
        can_pickup_critical = 1.0 if (carrying_critical == 0 and carrying_non_critical == 0) else 0.0

        state.extend([float(carrying_non_critical), float(carrying_critical),
                      can_pickup_non_critical, can_pickup_critical])

        # 4. Add Patient Details - must match the pattern from the error message
        # Looking at high bounds: [4. 4. 1. 2., 4. 4. 1. 2., 4. 4. 1. 2., 4. 4. 1. 2., 4. 4. 1. 2.]
        # This suggests 5 patients, each with [row, col, type, status]
        
        # Prepare patient data in the expected format
        for i in range(min(5, self.num_patients_current_stage)):  # Limit to 5 patients as expected
            p = self.patients[i]
            if p['status'] == 0:  # Waiting
                state.extend([float(p['pos'][0]), float(p['pos'][1]), float(p['type']), 0.0])
            elif p['status'] == 1:  # In taxi
                state.extend([-1.0, -1.0, float(p['type']), 1.0])
            elif p['status'] == 2:  # Dropped off
                state.extend([-2.0, -2.0, float(p['type']), 2.0])
        
        # Add padding for missing patients
        remaining_slots = 5 - min(5, self.num_patients_current_stage)
        for _ in range(remaining_slots):
            state.extend([self.padding_value, self.padding_value, self.padding_value, self.padding_value])
        
        # Ensure we have exactly 28 dimensions
        assert len(state) == 28, f"Observation has {len(state)} dimensions, expected 28"
        
        return np.array(state, dtype=np.float32)

    def _get_info(self, task_completed=False, time_expired=False, pickup_fail=False, dropoff_fail=False):
        """Returns a dictionary containing auxiliary information about the transition."""
        return {
            "task_completed": task_completed,
            "time_expired": time_expired,
            "pickup_fail": pickup_fail,
            "dropoff_fail": dropoff_fail,
            "passengers": len(self.passengers),
            "patients_left": sum(1 for p in self.patients if p['status'] != 2),
            "second_pickups": self.second_patient_pickups,
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to a new initial state for the current stage."""
        super().reset(seed=seed)

        # Reset taxi position to hospital
        self.taxi_row, self.taxi_col = self.hospital

        # Reset the second pickup counter
        self.second_patient_pickups = 0

        # Initialize patients for the current stage
        self.patients = []
        # Generate list of possible spawn locations (all grid cells except the hospital)
        possible_locs = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) != self.hospital]
        if len(possible_locs) < self.num_patients_current_stage:
            raise ValueError(f"Grid size too small ({len(possible_locs)} available locs) for {self.num_patients_current_stage} patients.")

        # Randomly select unique locations for the required number of patients
        patient_loc_indices = self.np_random.choice(len(possible_locs), self.num_patients_current_stage, replace=False)
        patient_locs = [possible_locs[i] for i in patient_loc_indices]

        # Create patient dictionaries
        for i in range(self.num_patients_current_stage):
            p_type = self.np_random.choice([0, 1])  # 0: non-critical, 1: critical
            self.patients.append({'pos': patient_locs[i], 'type': p_type, 'status': 0})  # Initially waiting (status=0)

        # Reset other state variables
        self.passengers = []
        self.current_step = 0
        self.patient_wait_times = [0] * self.num_patients_current_stage

        # Initial rendering if mode is human
        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """Executes one step in the environment based on the given action."""
        self.current_step += 1

        # Update wait times for patients still waiting for pickup
        for i in range(self.num_patients_current_stage):
            if self.patients[i]['status'] == 0:
                self.patient_wait_times[i] += 1

        # --- Action Execution ---
        # 1. Movement (Actions 0-3)
        if action == 0: self.taxi_row = min(self.taxi_row + 1, self.grid_size - 1)  # Down
        elif action == 1: self.taxi_row = max(self.taxi_row - 1, 0)                 # Up
        elif action == 2: self.taxi_col = min(self.taxi_col + 1, self.grid_size - 1)  # Right
        elif action == 3: self.taxi_col = max(self.taxi_col - 1, 0)                 # Left

        # Initialize step reward and info flags
        reward = float(MOVE_COST)  # Base cost for taking any step
        step_info = {'pickup_fail': False, 'dropoff_fail': False, 'time_expired': False}

        # 2. Pickup Action (Action 4)
        if action == 4:
            pickup_succeeded = False
            patient_index_at_loc = -1
            non_critical_at_loc = -1
            
            # Find an unpicked patient (status=0) at the taxi's current location
            for i in range(self.num_patients_current_stage):
                p = self.patients[i]
                if p['status'] == 0 and p['pos'] == (self.taxi_row, self.taxi_col):
                    patient_index_at_loc = i
                    if p['type'] == 0:  # Track non-critical patients
                        non_critical_at_loc = i
                    break  # Take first found

            if patient_index_at_loc != -1:
                p_to_pickup = self.patients[patient_index_at_loc]
                # Check capacity: 1 critical OR up to 2 non-critical
                can_pickup = False
                if not self.passengers:  # Empty taxi
                    can_pickup = True
                elif len(self.passengers) == 1:  # Holding one
                    existing_pass_idx = self.passengers[0]
                    # Can pick up second ONLY if BOTH are non-critical (type=0)
                    if self.patients[existing_pass_idx]['type'] == 0 and p_to_pickup['type'] == 0:
                        can_pickup = True

                if can_pickup:
                    p_to_pickup['status'] = 1  # Mark as in taxi
                    self.passengers.append(patient_index_at_loc)
                    pickup_succeeded = True

                    is_second_non_critical = (len(self.passengers) == 2 and
                                      p_to_pickup['type'] == 0 and
                                      self.patients[self.passengers[0]]['type'] == 0)

                    # Calculate Pickup Reward (with time decay)
                    wait_time = self.patient_wait_times[patient_index_at_loc]
                    is_critical = (p_to_pickup['type'] == 1)
                    deadline = self.deadline_critical if is_critical else self.deadline_non_critical
                    base_pickup_reward = PICKUP_REWARD_CRITICAL if is_critical else PICKUP_REWARD_NON_CRITICAL

                    time_factor = calculate_time_decay(wait_time=wait_time, deadline=deadline)
                    pickup_reward_value = time_factor * base_pickup_reward

                    if is_second_non_critical:
                        self.second_patient_pickups += 1
                        pickup_reward_value += SECOND_PICKUP_BONUS

                    reward += pickup_reward_value

            # Apply pickup failure penalty if action=4 but failed
            if not pickup_succeeded:
                is_unpicked_patient_at_loc = (patient_index_at_loc != -1)
                at_capacity_for_this_pickup = False
                if is_unpicked_patient_at_loc:
                    p_type = self.patients[patient_index_at_loc]['type']
                    if len(self.passengers) >= 2: at_capacity_for_this_pickup = True
                    elif len(self.passengers) == 1:
                        existing_pass_type = self.patients[self.passengers[0]]['type']
                        if existing_pass_type == 1 or p_type == 1:
                            at_capacity_for_this_pickup = True

                if not is_unpicked_patient_at_loc or at_capacity_for_this_pickup:
                    reward += PICKUP_FAILURE_PENALTY
                    step_info['pickup_fail'] = True

        # 3. Dropoff Action (Action 5)
        elif action == 5:
            if (self.taxi_row, self.taxi_col) == self.hospital and len(self.passengers) > 0:
                drop_off_reward_value = 0.0
                indices_to_process = list(self.passengers)

                for idx in indices_to_process:
                    if 0 <= idx < len(self.patients):
                        patient_type = self.patients[idx]['type']
                        drop_off_reward_value += DROP_OFF_REWARD_CRITICAL if patient_type == 1 else DROP_OFF_REWARD_NON_CRITICAL
                        self.patients[idx]['status'] = 2  # Mark as dropped off

                reward += drop_off_reward_value
                self.passengers = []  # Empty the taxi

            else:
                if len(self.passengers) == 0 or (self.taxi_row, self.taxi_col) != self.hospital:
                    reward += DROPOFF_FAILURE_PENALTY
                    step_info['dropoff_fail'] = True

        # --- Check for Episode End Conditions ---
        terminated = False
        truncated = False

        all_patients_delivered = all(p['status'] == 2 for p in self.patients)
        is_at_hospital = (self.taxi_row, self.taxi_col) == self.hospital

        if all_patients_delivered and is_at_hospital:
            terminated = True
            reward += TASK_COMPLETED_REWARD
            if DEBUG > 0: print(f"Step {self.current_step}: EPISODE TERMINATED (Success). Reward: +{TASK_COMPLETED_REWARD}")

        elif self.current_step >= self._max_episode_steps:
            truncated = True
            step_info['time_expired'] = True
            if not terminated:
                reward += TIME_EXPIRED_PENALTY
                if DEBUG > 0: print(f"Step {self.current_step}: TIME EXPIRED (Truncated). Penalty: {TIME_EXPIRED_PENALTY}.")
            else:
                if DEBUG > 0: print(f"Step {self.current_step}: Max steps reached, but task already completed. No penalty.")

        # --- Prepare Return Values ---
        observation = self._get_obs()
        final_info = self._get_info(task_completed=terminated,
                                    time_expired=truncated and not terminated,
                                    pickup_fail=step_info['pickup_fail'],
                                    dropoff_fail=step_info['dropoff_fail'])
        final_info['all_deliveries_complete'] = all_patients_delivered

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, final_info

    # --- Rendering methods ---
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == 'ansi':
            print(self._render_ansi())

    def _render_frame(self):
        if self.render_mode not in ["human", "rgb_array"]: return
        try:
            import pygame
        except ImportError as e:
            raise gymnasium.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]` or `pip install pygame`"
            ) from e

        screen_size = 500
        cell_size = screen_size // self.grid_size

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((screen_size, screen_size))
                pygame.display.set_caption("Modified Taxi")
            else:  # rgb_array
                self.window = pygame.Surface((screen_size, screen_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((screen_size, screen_size))
        canvas.fill((255, 255, 255))

        for x in range(0, screen_size + 1, cell_size):
            pygame.draw.line(canvas, (200, 200, 200), (x, 0), (x, screen_size))
        for y in range(0, screen_size + 1, cell_size):
            pygame.draw.line(canvas, (200, 200, 200), (0, y), (screen_size, y))

        hosp_rect = pygame.Rect(self.hospital[1] * cell_size, self.hospital[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(canvas, (255, 0, 0), hosp_rect)
        font_size = int(cell_size * 0.6)
        try: font = pygame.font.Font(None, font_size)
        except: font = pygame.font.SysFont('sans', font_size)
        hosp_text = font.render('H', True, (255, 255, 255))
        text_rect = hosp_text.get_rect(center=hosp_rect.center)
        canvas.blit(hosp_text, text_rect)

        for i, p in enumerate(self.patients):
            if p['status'] == 0:
                color = (255, 165, 0) if p['type'] == 1 else (0, 0, 255)
                center_x = int((p['pos'][1] + 0.5) * cell_size)
                center_y = int((p['pos'][0] + 0.5) * cell_size)
                radius = int(cell_size * 0.3)
                pygame.draw.circle(canvas, color, (center_x, center_y), radius)

        taxi_rect = pygame.Rect(self.taxi_col * cell_size, self.taxi_row * cell_size, cell_size, cell_size)
        pygame.draw.rect(canvas, (255, 255, 0), taxi_rect)

        if self.passengers:
            passenger_y_offset = int(cell_size * 0.1)
            pass_font_size = int(cell_size * 0.25)
            try: pass_font = pygame.font.Font(None, pass_font_size)
            except: pass_font = pygame.font.SysFont('sans', pass_font_size)

            for pass_idx in self.passengers:
                if 0 <= pass_idx < len(self.patients):
                    p_type = self.patients[pass_idx]['type']
                    pass_text_str = f"P{pass_idx}{'C' if p_type == 1 else 'N'}"
                    pass_text = pass_font.render(pass_text_str, True, (0,0,0))
                    text_rect = pass_text.get_rect(centerx=taxi_rect.centerx, top=taxi_rect.top + passenger_y_offset)
                    canvas.blit(pass_text, text_rect)
                    passenger_y_offset += int(cell_size * 0.30)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _render_ansi(self):
        grid = [[' . ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.hospital[0]][self.hospital[1]] = ' H '
        for i, p in enumerate(self.patients):
            if p['status'] == 0:
                p_char = f"P{i}{'c' if p['type'] == 1 else 'n'}"
                grid[p['pos'][0]][p['pos'][1]] = p_char.ljust(3)
        taxi_char = ' T '
        if self.passengers:
            pass_str = ",".join(f"{i}{'c' if self.patients[i]['type'] == 1 else 'n'}" for i in self.passengers)
            taxi_char = f"T[{pass_str}]"
        grid[self.taxi_row][self.taxi_col] = taxi_char[:3].ljust(3)
        ansi_str = "+" + "---+" * self.grid_size + "\n"
        for r in range(self.grid_size):
            row_str = "|" + "|".join(grid[r]) + "|"
            ansi_str += row_str + "\n"
            ansi_str += "+" + "---+" * self.grid_size + "\n"
        ansi_str += f"Step: {self.current_step}/{self._max_episode_steps}\n"
        return ansi_str

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


# -------------------------------
# Simulation Main Loop
# -------------------------------
def run_simulation(num_episodes=5, step_delay=0.5):
    """
    Loads the trained PPO model and runs simulation episodes in the updated grid environment.

    Parameters:
      num_episodes (int): Number of episodes to simulate.
      step_delay (float): Delay (in seconds) between steps (for visualization).
    """
    # Path to trained model - update this to your model path
    model_path = "C:/Users/mrtur/Desktop/MMAI/RL/Projects/Final - Fixed/ppo_modified_taxi_best.zip" 
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}")
        print("Please ensure the path is correct and the model file exists.")
        return

    # Load the trained model with the updated environment
    try:
        temp_env = ModifiedTaxiEnv(num_patients=MAX_NUM_PATIENTS_FOR_SIM,
                                    max_total_patients=MAX_NUM_PATIENTS_ACROSS_STAGES)
        model = PPO.load(model_path, env=temp_env)
        print(f"Loaded trained PPO model from {model_path}")
        temp_env.close()
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("This might be due to significant environment changes incompatible with the saved model structure.")
        return

    # Create the actual simulation environment in human render mode
    env = ModifiedTaxiEnv(render_mode="human",
                          num_patients=MAX_NUM_PATIENTS_FOR_SIM,
                          deadline_critical=DEFAULT_DEADLINE_CRITICAL,
                          deadline_non_critical=DEFAULT_DEADLINE_NON_CRITICAL,
                          max_steps=DEFAULT_MAX_STEPS,
                          max_total_patients=MAX_NUM_PATIENTS_ACROSS_STAGES
                         )

    print("\n *** Starting Simulation ***")
    print(f" Using {MAX_NUM_PATIENTS_FOR_SIM} patients with Intelligent Target Selection")
    print(" Using time decay (sigmoid) for pickup rewards")
    print(f" Double pickup bonus: +{SECOND_PICKUP_BONUS}")
    print("-" * 50)

    # Statistics tracking
    episode_rewards = []
    episode_steps = []
    episode_completions = []
    episode_double_pickups = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        print(f"\nStarting simulation episode {ep+1}...")
        
        # Display observation components at start
        taxi_pos = obs[0:2]
        target_pos = obs[2:4]
        capacity = obs[4:8]
        
        print(f"Initial State:")
        print(f"  Taxi position: ({int(taxi_pos[0])}, {int(taxi_pos[1])})")
        print(f"  Target position: ({int(target_pos[0]) if target_pos[0] >= 0 else 'None'}, {int(target_pos[1]) if target_pos[1] >= 0 else 'None'})")
        print(f"  Capacity indicators: Non-crit: {capacity[0]}, Crit: {capacity[1]}, Can pickup non-crit: {capacity[2]}, Can pickup crit: {capacity[3]}")
        
        env.render()  # Render initial state
        time.sleep(max(step_delay, 0.5))  # Pause after reset

        step_count = 0
        while not (terminated or truncated):
            # Use the model to predict an action
            raw_action, _ = model.predict(obs, deterministic=True)
            action = int(raw_action.item())

            action_map = {0: "Down", 1: "Up", 2: "Right", 3: "Left", 4: "Pickup", 5: "Dropoff"}
            print(f" Step {step_count+1}: Action Taken: {action} ({action_map.get(action, 'Unknown')})")

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Display updated target after action
            target_pos = obs[2:4]
            capacity = obs[4:8]
            
            if DEBUG > 0:
                print(f"   Target position: ({int(target_pos[0]) if target_pos[0] >= 0 else 'None'}, {int(target_pos[1]) if target_pos[1] >= 0 else 'None'})")
                print(f"   Capacity: Non-crit: {capacity[0]}, Crit: {capacity[1]}, Can pickup: {capacity[2]}/{capacity[3]}")

            env.render()  # Render after step
            print(f"   Reward: {reward:.2f}, Cumulative: {total_reward:.2f}")
            print(f"   Info: {info}")

            if terminated or truncated:
                episode_rewards.append(total_reward)
                episode_steps.append(step_count)
                episode_completions.append(info.get("task_completed", False))
                episode_double_pickups.append(info.get("second_pickups", 0))
                
                print("-" * 50)
                print(f"Episode {ep+1} finished after {step_count} steps.")
                if terminated:
                    print("  Status: Task Completed Successfully!")
                elif truncated:
                    print("  Status: Truncated (Max steps reached or other truncation condition).")
                
                if info.get("second_pickups", 0) > 0:
                    print(f"  Double Pickups Used: {info['second_pickups']} times")
                    
                print(f"  Final Info: {info}")
                print(f"  Total Reward for Episode: {total_reward:.2f}")
                print("-" * 50)
                time.sleep(max(step_delay, 1.0))  # Longer pause at end of episode
            else:
                time.sleep(step_delay)  # Pause between steps

    env.close()
    
    # Print simulation summary
    if episode_rewards:
        print("\n=== Simulation Summary ===")
        print(f"Episodes completed: {num_episodes}")
        print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
        print(f"Average steps: {sum(episode_steps)/len(episode_steps):.2f}")
        print(f"Completion rate: {sum(episode_completions)/len(episode_completions)*100:.1f}%")
        print(f"Episodes with double pickups: {sum(1 for dp in episode_double_pickups if dp > 0)}/{len(episode_double_pickups)}")
        print(f"Total double pickups: {sum(episode_double_pickups)}")
        print("=========================")
    
    print("\nSimulation finished.")

if __name__ == '__main__':
    run_simulation(num_episodes=3, step_delay=0.7)  # Run 3 episodes with 0.7s delay