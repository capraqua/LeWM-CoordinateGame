"""
Coordinate Game with Mines.

Agent and goal at random positions. Mines scattered randomly.
Agent can see mines within 1 step (8 directional sensors).
Stepping on a mine = -1 punishment.

State: [x, y, sensor_up, sensor_up_right, sensor_right, sensor_down_right,
        sensor_down, sensor_down_left, sensor_left, sensor_up_left]

Actions: up, down, left, right (discrete, step size 1).
"""

import torch
import random
from mpc_controller import MPCController

ACTIONS = {
    "up":    torch.tensor([0.0,  1.0]),
    "down":  torch.tensor([0.0, -1.0]),
    "left":  torch.tensor([-1.0, 0.0]),
    "right": torch.tensor([1.0,  0.0]),
}
ACTION_LIST = list(ACTIONS.values())
ACTION_NAMES = list(ACTIONS.keys())

# 8 neighbor offsets: up, up-right, right, down-right, down, down-left, left, up-left
SENSOR_OFFSETS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]


class CoordinateGame:
    """2D grid game with random mines. Agent must reach a random goal."""

    def __init__(self, grid_range=20, num_mines=10):
        self.grid_range = grid_range
        self.num_mines = num_mines
        self.mines = set()
        self.penalty = 0.0
        self.reset()

    def _random_pos(self):
        return (
            random.randint(-self.grid_range, self.grid_range),
            random.randint(-self.grid_range, self.grid_range),
        )

    def _compute_sensors(self):
        """Return 8 binary sensors for mines in neighboring cells."""
        x, y = int(self.pos[0].item()), int(self.pos[1].item())
        sensors = []
        for dx, dy in SENSOR_OFFSETS:
            sensors.append(1.0 if (x + dx, y + dy) in self.mines else 0.0)
        return torch.tensor(sensors)

    def _build_state(self):
        """Full state: [x, y, 8 sensors]."""
        return torch.cat([self.pos.clone(), self._compute_sensors()])

    def reset(self):
        sx, sy = self._random_pos()
        gx, gy = self._random_pos()
        while sx == gx and sy == gy:
            gx, gy = self._random_pos()

        self.pos = torch.tensor([float(sx), float(sy)])
        self._goal = torch.tensor([float(gx), float(gy)])
        self.penalty = 0.0

        # Place mines avoiding start and goal
        self.mines = set()
        while len(self.mines) < self.num_mines:
            mx, my = self._random_pos()
            if (mx, my) != (sx, sy) and (mx, my) != (gx, gy):
                self.mines.add((mx, my))

        return self._build_state()

    def step(self, action_vec):
        """Apply action, return (state_10d, reward, done)."""
        self.pos = self.pos + action_vec
        x, y = int(self.pos[0].item()), int(self.pos[1].item())

        # Mine check
        reward = 0.0
        if (x, y) in self.mines:
            reward = -1.0
            self.penalty += 1.0

        done = torch.norm(self.pos - self._goal).item() < 0.5
        return self._build_state(), reward, done

    @property
    def goal(self):
        return self._goal.clone()

    @property
    def goal_state_10d(self):
        """Goal as 10D state (goal position + all sensors zero = no mines at goal)."""
        return torch.cat([self._goal.clone(), torch.zeros(8)])


def continuous_to_discrete(action_vec):
    """Map continuous 2D action to nearest discrete action."""
    sims = [torch.dot(action_vec, a).item() for a in ACTION_LIST]
    idx = max(range(len(sims)), key=lambda i: sims[i])
    return ACTION_LIST[idx], ACTION_NAMES[idx]


def collect_game_data(num_episodes=100, max_steps=20, num_mines=10):
    """Play random episodes to collect transition data."""
    game = CoordinateGame(grid_range=20, num_mines=num_mines)
    states, actions, next_states, rewards = [], [], [], []

    for _ in range(num_episodes):
        state = game.reset()
        for _ in range(max_steps):
            action = random.choice(ACTION_LIST)
            next_state, reward, done = game.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            state = next_state
            if done:
                break

    return (torch.stack(states), torch.stack(actions),
            torch.stack(next_states), torch.tensor(rewards))


def render(state_10d, goal, mines, grid_range=20, path=None):
    """Render grid with mines and optional agent path."""
    r = grid_range + 1
    sx, sy = round(state_10d[0].item()), round(state_10d[1].item())
    gx, gy = round(goal[0].item()), round(goal[1].item())

    path_set = {}
    if path:
        for i, (px, py) in enumerate(path):
            path_set[(px, py)] = i

    rows = []
    for y in range(r, -r - 1, -1):
        row = ""
        for x in range(-r, r + 1):
            if x == sx and y == sy and x == gx and y == gy:
                row += "★"
            elif x == sx and y == sy:
                row += "A"
            elif x == gx and y == gy:
                row += "G"
            elif (x, y) in mines and path and (x, y) in path_set:
                row += "!"  # stepped on mine
            elif (x, y) in mines:
                row += "X"
            elif path and (x, y) in path_set:
                row += "○"
            elif x == 0 or y == 0:
                row += "·"
            else:
                row += " "
        rows.append(row)
    return "\n".join(rows)


def main():
    num_mines = 80
    game = CoordinateGame(grid_range=20, num_mines=num_mines)

    # State is now 10D (x, y + 8 sensors)
    controller = MPCController(
        state_dim=10,
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=5,
    )

    # Collect and train
    print(f"Collecting random game data ({num_mines} mines per episode)...")
    states, actions, next_states, rewards = collect_game_data(
        100, max_steps=15, num_mines=num_mines
    )
    print(f"  {len(states)} transitions collected")

    print("Training models...")
    controller.train_models(states, actions, next_states,
                            epochs=300, lr=1e-3)

    # Play episodes
    print("\n" + "=" * 50)
    print("PLAYING WITH MPC AGENT")
    print("=" * 50)

    wins = 0
    total_penalties = 0
    num_episodes = 5

    for ep in range(num_episodes):
        state = game.reset()
        goal = game.goal
        goal_10d = game.goal_state_10d
        print(f"\n--- Episode {ep+1} | Start: ({state[0]:.0f}, {state[1]:.0f}) "
              f"| Goal: ({goal[0]:.0f}, {goal[1]:.0f}) | Mines: {len(game.mines)} ---")
        print(render(state, goal, game.mines))

        ep_penalty = 0
        agent_path = [(int(state[0].item()), int(state[1].item()))]
        for step in range(80):
            raw_action = controller.step(state, goal_10d)
            action_vec, action_name = continuous_to_discrete(raw_action)
            state, reward, done = game.step(action_vec)
            agent_path.append((int(state[0].item()), int(state[1].item())))

            hit = " 💥 MINE!" if reward < 0 else ""
            if reward < 0:
                ep_penalty += 1

            sensors = state[2:].tolist()
            nearby = sum(1 for s in sensors if s > 0.5)
            print(f"  step {step+1}: {action_name:5s} → ({state[0]:.0f}, {state[1]:.0f}) "
                  f"| nearby mines: {nearby}{hit}")

            if done:
                print(f"  ★ Reached goal in {step+1} steps! (penalties: {ep_penalty})")
                print(render(state, goal, game.mines, path=agent_path))
                wins += 1
                break
        else:
            print(f"  ✗ Did not reach goal. Final: ({state[0]:.0f}, {state[1]:.0f})")
            print(render(state, goal, game.mines, path=agent_path))

        total_penalties += ep_penalty

    print(f"\nResults: {wins}/{num_episodes} reached goal | total mine hits: {total_penalties}")


if __name__ == "__main__":
    main()
