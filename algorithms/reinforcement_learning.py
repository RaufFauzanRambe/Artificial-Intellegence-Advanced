"""
Reinforcement Learning Module - Q-Learning Algorithm from Scratch.

Implements tabular Q-Learning with epsilon-greedy exploration, exponential
epsilon decay, and a configurable GridWorld environment for testing and
demonstration. No external dependencies beyond NumPy.

Example:
    >>> from reinforcement_learning import QLearningAgent, GridWorldEnv
    >>> env = GridWorldEnv(size=5)
    >>> agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=1.0)
    >>> agent.train(env, n_episodes=1000)
    >>> path = agent.get_optimal_path(env)
"""

import numpy as np


class GridWorldEnv:
    """A simple GridWorld environment for reinforcement learning.

    The agent starts at the top-left corner (0, 0) and must reach the goal
    at the bottom-right corner (size-1, size-1). Obstacles can be placed
    in the grid to create more complex navigation problems.

    Actions:
        0: Up    (row - 1)
        1: Down  (row + 1)
        2: Left  (col - 1)
        3: Right (col + 1)

    Rewards:
        -1.0 per step (encourages shortest paths)
        +10.0 for reaching the goal
        -5.0 for hitting an obstacle
        -2.0 for stepping outside the grid

    Attributes:
        size: Grid dimension (size x size).
        start_pos: Starting position (row, col).
        goal_pos: Goal position (row, col).
        obstacles: Set of (row, col) positions that are blocked.
        n_actions: Number of available actions (4).
        n_states: Total number of states (size * size).
        action_names: Human-readable action names.
    """

    # Action constants
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, size: int = 4, obstacles: list = None):
        """Initialize the GridWorld environment.

        Args:
            size: Grid dimension (size x size).
            obstacles: List of (row, col) tuples representing blocked cells.
        """
        self.size = size
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.obstacles = set(obstacles) if obstacles else set()
        self.n_actions = 4
        self.n_states = size * size
        self.action_names = ["Up", "Down", "Left", "Right"]
        self.agent_pos = None
        self.steps = 0
        self.max_steps = size * size * 4
        self.reset()

    def reset(self) -> int:
        """Reset the environment to the starting position.

        Returns:
            The initial state as a flat integer index.
        """
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._pos_to_state(self.agent_pos)

    def step(self, action: int) -> tuple:
        """Take a step in the environment.

        Args:
            action: Integer action code (0=Up, 1=Down, 2=Left, 3=Right).

        Returns:
            A tuple (next_state, reward, done, info) where:
                - next_state: Flat integer index of the new position.
                - reward: Reward received for this step.
                - done: True if the episode has ended.
                - info: Dictionary with additional information.
        """
        self.steps += 1
        row, col = self.agent_pos

        # Compute new position based on action
        if action == self.UP:
            new_row, new_col = row - 1, col
        elif action == self.DOWN:
            new_row, new_col = row + 1, col
        elif action == self.LEFT:
            new_row, new_col = row, col - 1
        elif action == self.RIGHT:
            new_row, new_col = row, col + 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check boundaries
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            reward = -2.0
            done = False
            info = {"result": "out_of_bounds"}
            return self._pos_to_state(self.agent_pos), reward, done, info

        new_pos = (new_row, new_col)

        # Check obstacles
        if new_pos in self.obstacles:
            reward = -5.0
            done = False
            info = {"result": "hit_obstacle"}
            return self._pos_to_state(self.agent_pos), reward, done, info

        # Move agent
        self.agent_pos = new_pos

        # Check if goal reached
        if new_pos == self.goal_pos:
            reward = 10.0
            done = True
            info = {"result": "goal_reached", "steps": self.steps}
        else:
            reward = -1.0
            done = False
            info = {"result": "moved"}

        # Timeout to prevent infinite loops
        if self.steps >= self.max_steps:
            done = True

        return self._pos_to_state(new_pos), reward, done, info

    def _pos_to_state(self, pos: tuple) -> int:
        """Convert a (row, col) position to a flat state index.

        Args:
            pos: Tuple of (row, col).

        Returns:
            Integer state index.
        """
        return pos[0] * self.size + pos[1]

    def state_to_pos(self, state: int) -> tuple:
        """Convert a flat state index to a (row, col) position.

        Args:
            state: Integer state index.

        Returns:
            Tuple of (row, col).
        """
        return divmod(state, self.size)

    def render(self) -> str:
        """Render the current grid state as a string.

        Returns:
            String representation of the grid.
                S = Start, G = Goal, X = Obstacle, A = Agent, . = Empty
        """
        grid = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row.append(" A ")
                elif (r, c) == self.start_pos:
                    row.append(" S ")
                elif (r, c) == self.goal_pos:
                    row.append(" G ")
                elif (r, c) in self.obstacles:
                    row.append(" X ")
                else:
                    row.append(" . ")
            grid.append("|" + "|".join(row) + "|")
        return "\n".join(grid)


class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration.

    Learns an optimal action-value function Q(s, a) through iterative updates:
        Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))

    Uses epsilon-greedy policy for action selection during training, with
    exponential decay of epsilon over episodes.

    Attributes:
        alpha: Learning rate (step size).
        gamma: Discount factor for future rewards.
        epsilon: Current exploration rate.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Multiplicative decay factor per episode.
        q_table: 2-D array of shape (n_states, n_actions).
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """Initialize the Q-Learning agent.

        Args:
            n_states: Number of discrete states in the environment.
            n_actions: Number of discrete actions available.
            alpha: Learning rate (0 < alpha <= 1).
            gamma: Discount factor (0 <= gamma <= 1).
            epsilon: Initial exploration probability.
            epsilon_min: Minimum exploration probability.
            epsilon_decay: Multiplicative decay per episode (epsilon *= decay).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state: int) -> int:
        """Select an action using epsilon-greedy policy.

        With probability epsilon, selects a random action (exploration).
        Otherwise, selects the action with the highest Q-value (exploitation).
        Ties are broken randomly.

        Args:
            state: Current state index.

        Returns:
            Selected action as an integer.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Break ties randomly among actions with max Q-value
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ):
        """Update Q-table using the Q-Learning update rule.

        Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
        where target = reward + gamma * max_a' Q(s', a') for non-terminal states
        or target = reward for terminal states.

        Args:
            state: Current state index.
            action: Action taken.
            reward: Reward received.
            next_state: Next state index.
            done: Whether the episode ended after this step.
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def train(
        self,
        env: GridWorldEnv,
        n_episodes: int = 1000,
        verbose: bool = True,
    ) -> dict:
        """Train the agent on the given environment.

        Args:
            env: GridWorldEnv instance to train on.
            n_episodes: Number of training episodes.
            verbose: If True, print progress every 100 episodes.

        Returns:
            Dictionary with training statistics:
                - rewards_per_episode: List of total rewards per episode.
                - steps_per_episode: List of steps per episode.
                - epsilon_history: List of epsilon values per episode.
        """
        rewards_history = []
        steps_history = []
        epsilon_history = []

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            steps = 0

            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

            rewards_history.append(total_reward)
            steps_history.append(steps)
            epsilon_history.append(self.epsilon)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                avg_steps = np.mean(steps_history[-100:])
                print(f"  Episode {episode + 1:4d}/{n_episodes} | "
                      f"Avg Reward (last 100): {avg_reward:7.2f} | "
                      f"Avg Steps: {avg_steps:5.1f} | "
                      f"Epsilon: {self.epsilon:.4f}")

        return {
            "rewards_per_episode": rewards_history,
            "steps_per_episode": steps_history,
            "epsilon_history": epsilon_history,
        }

    def get_q_table(self) -> np.ndarray:
        """Return the learned Q-table.

        Returns:
            Q-table of shape (n_states, n_actions).
        """
        return self.q_table.copy()

    def get_optimal_path(self, env: GridWorldEnv) -> list:
        """Extract the optimal path from start to goal using the learned policy.

        Follows a greedy policy (always picks the best action) from the start
        state. Stops when reaching the goal or after max steps.

        Args:
            env: GridWorldEnv instance.

        Returns:
            List of (row, col) positions representing the optimal path.
        """
        state = env.reset()
        path = [env.state_to_pos(state)]
        visited = set()

        for _ in range(env.max_steps):
            if state in visited:
                break  # Avoid infinite loops
            visited.add(state)

            action = np.argmax(self.q_table[state])
            next_state, _, done, _ = env.step(action)
            path.append(env.state_to_pos(next_state))
            state = next_state

            if done:
                break

        return path

    def print_policy(self, env: GridWorldEnv):
        """Print the learned policy as a grid of action arrows.

        Args:
            env: GridWorldEnv instance.
        """
        arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        print("\nLearned Policy:")
        for r in range(env.size):
            row = []
            for c in range(env.size):
                state = env._pos_to_state((r, c))
                if (r, c) in env.obstacles:
                    row.append(" X ")
                elif (r, c) == env.goal_pos:
                    row.append(" G ")
                else:
                    best_action = np.argmax(self.q_table[state])
                    row.append(f" {arrows[best_action]} ")
            print("|" + "|".join(row) + "|")


if __name__ == "__main__":
    print("=" * 60)
    print("Q-Learning Agent - GridWorld Demo (4x4, no obstacles)")
    print("=" * 60)

    # --- Simple 4x4 GridWorld ---
    env = GridWorldEnv(size=4)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    print(f"Grid size: {env.size}x{env.size}")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    print(f"States: {env.n_states}, Actions: {env.n_actions}")
    print(f"\nInitial Grid:")
    print(env.render())
    print()

    stats = agent.train(env, n_episodes=500, verbose=True)

    # Show results
    print("\nOptimal Path:")
    path = agent.get_optimal_path(env)
    for i, pos in enumerate(path):
        marker = " -> " if i < len(path) - 1 else ""
        print(f"  Step {i}: {pos}{marker}")

    agent.print_policy(env)

    # --- GridWorld with obstacles ---
    print("\n" + "=" * 60)
    print("Q-Learning Agent - GridWorld Demo (5x5, with obstacles)")
    print("=" * 60)

    obstacles_5x5 = [(0, 2), (1, 2), (2, 2), (3, 2), (1, 0)]
    env2 = GridWorldEnv(size=5, obstacles=obstacles_5x5)

    agent2 = QLearningAgent(
        n_states=env2.n_states,
        n_actions=env2.n_actions,
        alpha=0.15,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    print(f"Grid size: {env2.size}x{env2.size}")
    print(f"Obstacles: {sorted(env2.obstacles)}")
    print(f"\nGrid Layout:")
    print(env2.render())
    print()

    agent2.train(env2, n_episodes=1000, verbose=True)

    print("\nOptimal Path:")
    path2 = agent2.get_optimal_path(env2)
    for i, pos in enumerate(path2):
        marker = " -> " if i < len(path2) - 1 else ""
        print(f"  Step {i}: {pos}{marker}")

    agent2.print_policy(env2)

    # Training statistics
    print("\n" + "=" * 60)
    print("Training Statistics Summary")
    print("=" * 60)
    print(f"4x4 Grid - Final epsilon: {agent.epsilon:.6f}")
    print(f"5x5 Grid - Final epsilon: {agent2.epsilon:.6f}")
