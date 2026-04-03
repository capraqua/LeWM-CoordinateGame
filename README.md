# LeWM Coordinate Game

A JEPA-based world model (LeWorldModel) controlling an agent in a 2D mine-filled grid, using Model Predictive Control with CEM planning in latent space.

Based on: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture" (Maes, Le Lidec et al., 2026)

## How It Works

1. An **Encoder** maps the 10D state (x, y + 8 mine sensors) to a 64D latent embedding, with a BatchNorm projection head
2. A **Predictor** with residual connection predicts the next latent: `z_{t+1} = z_t + f(z_t, a_t)`
3. Training loss: `MSE(z_pred, z_target) + λ * SIGReg(Z)` — SIGReg prevents representation collapse by pushing latents toward an isotropic Gaussian distribution
4. **CEM planner** samples action sequences, rolls them out in latent space, and refines toward sequences that minimize latent distance to the goal
5. **MPC loop**: encode current state, plan via CEM, execute first action, replan

## Coordinate Game

- 41x41 grid (range ±20), 80 random mines
- Agent sees mines within 1 step via 8 directional sensors
- Actions: up, down, left, right (discrete)
- Stepping on a mine = -1 penalty
- Goal: reach a random target position

## Files

| File | Description |
|------|-------------|
| `mpc_controller.py` | LeWorldModel (Encoder, Predictor, SIGReg), CEM planner, MPC controller |
| `coordinate_game.py` | Grid game environment, data collection, ASCII rendering, main loop |

## Quick Start

```bash
pip install torch
python coordinate_game.py
```