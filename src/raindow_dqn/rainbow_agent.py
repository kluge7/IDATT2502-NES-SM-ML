import hyperparameters as hp
import torch
import torch.optim as optim
from prioritized_replay_buffer import PrioritizedReplayBuffer, Transition
from q_network import QNetwork


class RainbowAgent:
    def __init__(self, state_shape, action_size, device):
        self.device = device
        self.action_size = action_size
        self.gamma = hp.GAMMA
        self.batch_size = hp.BATCH_SIZE
        self.n_steps = hp.N_STEPS  # For multi-step learning
        self.target_update_frequency = hp.TARGET_UPDATE_FREQUENCY
        self.num_atoms = hp.NUM_ATOMS  # For distributional RL
        self.Vmin = hp.V_MIN
        self.Vmax = hp.V_MAX
        self.delta_z = (self.Vmax - self.Vmin) / (self.num_atoms - 1)
        self.support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms).to(
            self.device
        )

        self.policy_net = QNetwork(
            state_shape[0],
            action_size,
            num_atoms=self.num_atoms,
            noisy=True,
            dueling=True,
        ).to(device)
        self.target_net = QNetwork(
            state_shape[0],
            action_size,
            num_atoms=self.num_atoms,
            noisy=True,
            dueling=True,
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hp.LR)
        self.memory = PrioritizedReplayBuffer(hp.REPLAY_MEMORY_SIZE, hp.ALPHA)

        self.beta = hp.BETA_START
        self.beta_increment = (1.0 - hp.BETA_START) / hp.BETA_FRAMES
        self.steps_done = 0

    def select_action(self, state):
        """Selects an action using the policy network with Noisy Layers."""
        with torch.no_grad():
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            # No epsilon-greedy needed; NoisyNet handles exploration
            q_values = self.policy_net(state)
            action = q_values.mul(self.support).sum(2).max(1)[1].item()
            return action

    def update(self):
        """Performs a single training step using Rainbow DQN components."""
        if len(self.memory) < self.batch_size:
            return

        self.beta = min(1.0, self.beta + self.beta_increment)

        transitions, indices, weights = self.memory.sample(self.batch_size, self.beta)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(
            batch.action, device=self.device, dtype=torch.long
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float32
        )
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

        # Compute projected distribution (Distributional RL)
        with torch.no_grad():
            next_dist = self.target_net(next_state_batch)
            next_actions = next_dist.mul(self.support).sum(2).max(1)[1]
            next_dist = next_dist[range(self.batch_size), next_actions]

            t_z = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * (
                self.gamma**self.n_steps
            ) * self.support.unsqueeze(0)
            t_z = t_z.clamp(self.Vmin, self.Vmax)
            b = (t_z - self.Vmin) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros(self.batch_size, self.num_atoms, device=self.device)
            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.num_atoms, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .to(self.device)
            )

            m.view(-1).index_add_(
                0,
                (l + offset).view(-1),
                (next_dist * (u.float() - b)).view(-1),
            )
            m.view(-1).index_add_(
                0,
                (u + offset).view(-1),
                (next_dist * (b - l.float())).view(-1),
            )

        dist = self.policy_net(state_batch)
        log_p = torch.log(dist[range(self.batch_size), action_batch.squeeze(1)])
        loss = -(log_p * m).sum(1)
        loss = (loss * weights.to(self.device)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities in PER
        priorities = loss.detach().cpu().abs().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)

        # Reset noise in Noisy Layers
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def update_target_network(self):
        """Soft update of the target network parameters."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """Saves the policy network's state dictionary."""
        torch.save(self.policy_net.state_dict(), hp.MODEL_SAVE_PATH)
        print(f"Model saved to {hp.MODEL_SAVE_PATH}")

    def load_model(self, path):
        """Loads the policy network's state dictionary and updates the target network."""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path} on {self.device}")
