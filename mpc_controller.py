"""
LeWorldModel (LeWM) adapted for the coordinate game.

Based on: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
Architecture" (Maes, Le Lidec et al., 2026)

Core idea:
  1. Encoder maps state → latent embedding z
  2. Predictor predicts next latent: z_hat_{t+1} = pred(z_t, a_t)
  3. Loss = MSE(z_hat, z_target) + λ * SIGReg(Z)
     SIGReg prevents representation collapse by enforcing
     Gaussian-distributed latent embeddings.
  4. Planning via CEM in latent space: find action sequence that
     minimizes ||z_predicted_final - z_goal||²
  5. MPC: execute first action, replan from new observation.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Maps raw state to latent embedding."""

    def __init__(self, state_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Projection with BatchNorm (as in LeWM paper)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, state):
        h = self.net(state)
        if h.dim() == 1:
            h = h.unsqueeze(0)
            return self.proj(h).squeeze(0)
        return self.proj(h)


class Predictor(nn.Module):
    """Predicts next latent from current latent + action."""

    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, action):
        return z + self.net(torch.cat([z, action], dim=-1))


def sigreg(Z, num_projections=256):
    """
    Sketched-Isotropic-Gaussian Regularizer (SIGReg).

    Projects embeddings onto random directions and measures
    deviation from Gaussian via Epps-Pulley test statistic.
    Encourages latent embeddings to be isotropic Gaussian.

    Args:
        Z: (N, D) tensor of latent embeddings
        num_projections: number of random projection directions
    Returns:
        scalar regularization loss
    """
    if Z.shape[0] < 4:
        return torch.tensor(0.0, device=Z.device)

    D = Z.shape[-1]
    device = Z.device

    # Random unit-norm projection directions
    U = torch.randn(D, num_projections, device=device)
    U = U / U.norm(dim=0, keepdim=True)

    # Project: (N, M)
    H = Z @ U

    # Standardize each projection
    H = (H - H.mean(dim=0, keepdim=True)) / (H.std(dim=0, keepdim=True) + 1e-8)

    # Epps-Pulley test: compare empirical characteristic function
    # to standard Gaussian characteristic function
    N = H.shape[0]
    # Quadrature points
    num_knots = 20
    t = torch.linspace(0.2, 4.0, num_knots, device=device)

    # Empirical characteristic function: phi_N(t) = mean(exp(i*t*h))
    # For each projection m and knot k: (N, M, K)
    Ht = H.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)  # (N, M, K)
    ecf_real = torch.cos(Ht).mean(dim=0)  # (M, K)
    ecf_imag = torch.sin(Ht).mean(dim=0)  # (M, K)

    # Target: standard Gaussian cf: phi_0(t) = exp(-t²/2)
    gcf = torch.exp(-0.5 * t ** 2)  # (K,)

    # Weighted squared difference
    weight = torch.exp(-0.5 * t ** 2)  # Gaussian weighting
    diff_real = (ecf_real - gcf.unsqueeze(0)) ** 2
    diff_imag = ecf_imag ** 2
    test_stat = ((diff_real + diff_imag) * weight.unsqueeze(0)).mean(dim=-1)

    return test_stat.mean()


class LeWorldModel(nn.Module):
    """
    Joint-Embedding Predictive Architecture for the coordinate game.

    Encoder: state → latent z
    Predictor: (z_t, a_t) → z_hat_{t+1}
    Loss: MSE prediction + λ * SIGReg
    """

    def __init__(self, state_dim, action_dim, latent_dim=64,
                 hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(state_dim, latent_dim, hidden_dim)
        self.predictor = Predictor(latent_dim, action_dim, hidden_dim)
        self.latent_dim = latent_dim

    def encode(self, state):
        return self.encoder(state)

    def predict(self, z, action):
        return self.predictor(z, action)

    def rollout(self, z0, actions):
        """Roll out action sequence in latent space.
        Args:
            z0: (B, D) or (D,) initial latent
            actions: (B, H, A) or (H, A) action sequence
        Returns:
            z_final: (B, D) or (D,) final predicted latent
        """
        z = z0
        if actions.dim() == 2:
            # Single trajectory
            for t in range(actions.shape[0]):
                z = self.predictor(z.unsqueeze(0), actions[t].unsqueeze(0)).squeeze(0)
            return z
        # Batched
        for t in range(actions.shape[1]):
            z = self.predictor(z, actions[:, t])
        return z


class CEMPlanner:
    """
    Cross-Entropy Method planner in latent space.

    Samples action sequences, rolls them out through the predictor,
    scores by latent distance to goal, keeps elites, refines.
    """

    def __init__(self, action_dim, horizon=5, num_samples=200,
                 num_elites=20, num_iters=10,
                 action_low=-1.0, action_high=1.0):
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iters = num_iters
        self.action_low = action_low
        self.action_high = action_high

    def plan(self, z_start, z_goal, model):
        """
        Find action sequence that moves z_start toward z_goal.
        Returns first action of the best plan.
        """
        device = z_start.device
        H = self.horizon
        A = self.action_dim
        N = self.num_samples
        K = self.num_elites

        # Initialize CEM distribution
        mean = torch.zeros(H, A, device=device)
        std = torch.ones(H, A, device=device)

        best_action_seq = mean.clone()
        best_cost = float('inf')

        with torch.no_grad():
            for _ in range(self.num_iters):
                # Sample action sequences
                noise = torch.randn(N, H, A, device=device)
                actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise)
                actions = actions.clamp(self.action_low, self.action_high)

                # Rollout each sequence
                z = z_start.unsqueeze(0).expand(N, -1)
                for t in range(H):
                    z = model.predict(z, actions[:, t])

                # Cost: latent distance to goal
                costs = ((z - z_goal.unsqueeze(0)) ** 2).sum(dim=-1)

                # Select elites
                elite_idx = torch.argsort(costs)[:K]
                elites = actions[elite_idx]

                # Track best
                if costs[elite_idx[0]] < best_cost:
                    best_cost = costs[elite_idx[0]].item()
                    best_action_seq = elites[0]

                # Update distribution
                mean = elites.mean(dim=0)
                std = elites.std(dim=0).clamp(min=0.01)

        return best_action_seq[0]


class MPCController:
    """
    LeWorldModel MPC controller.

    Trains encoder + predictor with MSE + SIGReg.
    Plans via CEM in latent space.
    Executes first action, replans each step.
    """

    def __init__(self, state_dim, action_dim, latent_dim=64,
                 hidden_dim=128, horizon=5, action_low=-1.0,
                 action_high=1.0, device="cpu"):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = LeWorldModel(
            state_dim, action_dim, latent_dim, hidden_dim
        ).to(self.device)

        self.planner = CEMPlanner(
            action_dim=action_dim,
            horizon=horizon,
            action_low=action_low,
            action_high=action_high,
        )

    def step(self, state, goal):
        """Encode state and goal, plan in latent space, return action."""
        state = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device)
        goal = torch.as_tensor(goal, dtype=torch.float32,
                               device=self.device)

        self.model.eval()
        with torch.no_grad():
            z_state = self.model.encode(state)
            z_goal = self.model.encode(goal)

        action = self.planner.plan(z_state, z_goal, self.model)

        return action

    def train_models(self, states, actions, next_states,
                     epochs=100, lr=1e-3, sigreg_lambda=0.1, **_kw):
        """
        Train LeWorldModel: MSE prediction + SIGReg regularization.
        """
        states = torch.as_tensor(states, dtype=torch.float32,
                                 device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32,
                                  device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32,
                                      device=self.device)

        print("Training LeWorldModel (encoder + predictor)...")
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        n = states.shape[0]
        batch_size = min(256, n)

        for epoch in range(epochs):
            # Random batch
            idx = torch.randint(0, n, (batch_size,), device=self.device)
            s_batch = states[idx]
            a_batch = actions[idx]
            ns_batch = next_states[idx]

            # Encode current and next states
            z_curr = self.model.encode(s_batch)
            z_next = self.model.encode(ns_batch)

            # Predict next latent
            z_pred = self.model.predict(z_curr, a_batch)

            # Prediction loss (MSE)
            pred_loss = nn.functional.mse_loss(z_pred, z_next)

            # SIGReg: enforce Gaussian latent distribution
            reg_loss = sigreg(z_curr)

            loss = pred_loss + sigreg_lambda * reg_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (epoch + 1) % 25 == 0:
                print(f"  epoch {epoch+1}/{epochs}, "
                      f"pred={pred_loss.item():.6f}, "
                      f"sigreg={reg_loss.item():.6f}")
