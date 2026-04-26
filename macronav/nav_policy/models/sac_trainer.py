import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.nav import PolicyNet, QNet
from utils.misc import calcu_spl, load_models, save_models, write_to_tb


class SACTrainer:
    def __init__(self, train_param, device, local_device):
        self.train_param = train_param
        self.device = device
        self.local_device = local_device

        self.global_policy_net = PolicyNet(train_param.CONFIG_DICT).to(device)
        self.global_q_net1 = QNet(train_param.CONFIG_DICT).to(device)
        self.global_q_net2 = QNet(train_param.CONFIG_DICT).to(device)

        self.log_alpha = torch.FloatTensor([-2]).to(device)
        self.log_alpha.requires_grad = True

        self.global_target_q_net1 = QNet(train_param.CONFIG_DICT).to(device)
        self.global_target_q_net2 = QNet(train_param.CONFIG_DICT).to(device)
        self.global_target_q_net1.load_state_dict(self.global_q_net1.state_dict())
        self.global_target_q_net2.load_state_dict(self.global_q_net2.state_dict())
        self.global_target_q_net1.eval()
        self.global_target_q_net2.eval()

        self.global_policy_optimizer = optim.Adam(self.global_policy_net.parameters(), lr=train_param.LR_POLICY_NET)
        self.global_q_net1_optimizer = optim.Adam(self.global_q_net1.parameters(), lr=train_param.LR_Q_NET)
        self.global_q_net2_optimizer = optim.Adam(self.global_q_net2.parameters(), lr=train_param.LR_Q_NET)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=train_param.LR_ALPHA)

        self.policy_lr_decay = optim.lr_scheduler.StepLR(
            self.global_policy_optimizer, step_size=train_param.DECAY_STEP, gamma=0.96
        )
        self.q_net1_lr_decay = optim.lr_scheduler.StepLR(
            self.global_q_net1_optimizer, step_size=train_param.DECAY_STEP, gamma=0.96
        )
        self.q_net2_lr_decay = optim.lr_scheduler.StepLR(
            self.global_q_net2_optimizer, step_size=train_param.DECAY_STEP, gamma=0.96
        )
        self.log_alpha_lr_decay = optim.lr_scheduler.StepLR(
            self.log_alpha_optimizer, step_size=train_param.DECAY_STEP, gamma=0.96
        )

        self.models = {
            "policy_model": self.global_policy_net,
            "q_net1_model": self.global_q_net1,
            "q_net2_model": self.global_q_net2,
            "log_alpha": self.log_alpha,
            "policy_optimizer": self.global_policy_optimizer,
            "q_net1_optimizer": self.global_q_net1_optimizer,
            "q_net2_optimizer": self.global_q_net2_optimizer,
            "log_alpha_optimizer": self.log_alpha_optimizer,
            "episode": 0,
            "samples": 0,
            "policy_lr_decay": self.policy_lr_decay,
            "q_net1_lr_decay": self.q_net1_lr_decay,
            "q_net2_lr_decay": self.q_net2_lr_decay,
            "log_alpha_lr_decay": self.log_alpha_lr_decay,
        }

        self.entropy_target = train_param.ENTROPY_WEIGHT * (-np.log(1 / train_param.K_SIZE))

        self.dp_policy_net = nn.DataParallel(self.global_policy_net)
        self.dp_q_net1 = nn.DataParallel(self.global_q_net1)
        self.dp_q_net2 = nn.DataParallel(self.global_q_net2)
        self.dp_target_q_net1 = nn.DataParallel(self.global_target_q_net1)
        self.dp_target_q_net2 = nn.DataParallel(self.global_target_q_net2)

        self.train_log_data = []
        self.max_episodic_reward = -np.inf
        self.start_train_flag = False
        self.last_ckpt_save_step = 0
        self.last_perf_data = None
        self.log_counter = 0
        self.target_q_update_counter = 1

    def maybe_load_checkpoint(self, logger, runtime_state):
        if not self.train_param.LOAD_MODEL:
            return

        ckpt_path = (
            self.train_param.CKPT_PATH
            if self.train_param.CKPT_PATH
            else os.path.join(self.train_param.MODEL_PATH, "checkpoint_best.pth")
        )

        if not os.path.exists(ckpt_path):
            logger.warning("No checkpoint found, training from scratch")
            return

        logger.info(f"Loading model from {ckpt_path}")
        load_models(ckpt_path, self.models)
        runtime_state.curr_episode = self.models["curr_episode"]
        runtime_state.curr_samples = self.models["curr_samples"]
        logger.info(
            f"curr_episode set to {runtime_state.curr_episode}, curr_samples set to {runtime_state.curr_samples}"
        )

    def build_worker_weights(self):
        weights_set = []
        if self.device != self.local_device:
            policy_weights = self.global_policy_net.to(self.local_device).state_dict()
            q_net1_weights = self.global_q_net1.to(self.local_device).state_dict()
            self.global_policy_net.to(self.device)
            self.global_q_net1.to(self.device)
        else:
            policy_weights = self.global_policy_net.to(self.local_device).state_dict()
            q_net1_weights = self.global_q_net1.to(self.local_device).state_dict()

        weights_set.append(policy_weights)
        weights_set.append(q_net1_weights)
        return copy.deepcopy(weights_set)

    def replay_ready(self, runtime_state):
        return len(runtime_state.replay_buffer["done"]) >= self.train_param.REPLAY_BUFFER_MIN_SAMPLE

    def maybe_log_training_start(self, logger, runtime_state):
        if self.start_train_flag:
            return
        logger.info(f"\r\nStart training. Collected {len(runtime_state.replay_buffer['done'])} samples")
        self.start_train_flag = True

    def run_iteration(self, runtime_state, metric_names):
        last_stats = None
        for _ in range(self.train_param.GRADIENT_STEPS):
            batch = runtime_state.sample_batch(
                self.train_param.CONFIG_DICT,
                self.device,
            )
            last_stats = self._train_single_batch(batch, runtime_state)

        self.policy_lr_decay.step()
        self.q_net1_lr_decay.step()
        self.q_net2_lr_decay.step()
        self._maybe_update_target_networks()

        perf_data = self._collect_perf_data(runtime_state, metric_names)
        curr_log_data = [
            last_stats["reward_mean"],
            last_stats["value_prime_mean"],
            last_stats["policy_loss"],
            last_stats["q1_loss"],
            last_stats["entropy_mean"],
            last_stats["policy_grad_norm"],
            last_stats["q_grad_norm"],
            last_stats["log_alpha"],
            last_stats["alpha_loss"],
            *perf_data,
        ]
        self.train_log_data.append(curr_log_data)

    def maybe_flush_logs(self, runtime_state, log_writer):
        if (runtime_state.curr_samples - self.log_counter) < self.train_param.SUMMARY_WINDOW:
            return

        self.log_counter = runtime_state.curr_samples
        write_to_tb(log_writer, self.train_log_data, runtime_state.curr_samples)
        self.train_log_data = []
        runtime_state.reset_perf_metrics()

    def maybe_save_checkpoints(self, runtime_state):
        if runtime_state.curr_episode % self.train_param.MODEL_SAVE_FREQ == 0:
            self.models["episode"] = runtime_state.curr_episode
            self.models["log_alpha"] = self.log_alpha
            save_models(self.models, os.path.join(self.train_param.MODEL_PATH, "checkpoint_latest.pth"))

        if runtime_state.curr_samples - self.last_ckpt_save_step < 50000:
            return

        self.last_ckpt_save_step = runtime_state.curr_samples
        self.models["curr_episode"] = runtime_state.curr_episode
        self.models["curr_samples"] = runtime_state.curr_samples
        self.models["log_alpha"] = self.log_alpha
        ckpt_filename = f"checkpoint_{runtime_state.curr_samples // 1000}k.pth"
        save_models(self.models, os.path.join(self.train_param.MODEL_PATH, ckpt_filename))

    def _train_single_batch(self, batch, runtime_state):
        if batch["is_sequence"]:
            return self._train_sequence_batch(batch, runtime_state)
        return self._train_transition_batch(batch, runtime_state)

    def _reset_model_state(self, model):
        if hasattr(model, "reset_recurrent_state"):
            model.reset_recurrent_state()

    def _sequence_step_input(self, sequence_batch, step_idx):
        return tuple(None if tensor is None else tensor[:, step_idx] for tensor in sequence_batch)

    def _masked_mean(self, tensor, valid_mask):
        if valid_mask is None:
            return tensor.mean()
        denom = torch.clamp(valid_mask.sum(), min=1.0)
        return (tensor * valid_mask).sum() / denom

    def _forward_policy_sequence(self, sequence_batch, *, no_grad=False):
        self._reset_model_state(self.global_policy_net)
        logps = []
        grad_context = torch.no_grad() if no_grad else torch.enable_grad()
        with grad_context:
            for step_idx in range(sequence_batch[0].size(1)):
                logps.append(self.global_policy_net(self._sequence_step_input(sequence_batch, step_idx)))
        return torch.stack(logps, dim=1)

    def _forward_q_sequence(self, model, sequence_batch, *, no_grad=False):
        self._reset_model_state(model)
        q_values = []
        attention_weights = None
        grad_context = torch.no_grad() if no_grad else torch.enable_grad()
        with grad_context:
            for step_idx in range(sequence_batch[0].size(1)):
                q_value, attention_weights = model(self._sequence_step_input(sequence_batch, step_idx))
                q_values.append(q_value)
        return torch.stack(q_values, dim=1), attention_weights

    def _train_transition_batch(self, batch, runtime_state):
        state_batch = batch["state_batch"]
        next_state_batch = batch["next_state_batch"]
        q_state_batch = batch["q_state_batch"]
        next_q_state_batch = batch["next_q_state_batch"]
        action_batch = batch["action_batch"]
        reward_batch = batch["reward_batch"]
        done_batch = batch["done_batch"]

        with torch.no_grad():
            q_values1, _ = self.dp_q_net1(q_state_batch)
            q_values2, _ = self.dp_q_net2(q_state_batch)
            q_values = torch.min(q_values1, q_values2)

        logp = self.dp_policy_net(state_batch)
        policy_loss = torch.sum(
            logp.exp().unsqueeze(2) * (self.log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach()),
            dim=1,
        ).mean()
        self.global_policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.global_policy_net.parameters(), max_norm=10, norm_type=2)
        self.global_policy_optimizer.step()

        with torch.no_grad():
            next_logp = self.dp_policy_net(next_state_batch)
            next_q_values1, _ = self.dp_target_q_net1(next_q_state_batch)
            next_q_values2, _ = self.dp_target_q_net2(next_q_state_batch)
            next_q_values = torch.min(next_q_values1, next_q_values2)
            value_prime_batch = torch.sum(
                next_logp.unsqueeze(2).exp() * (next_q_values - self.log_alpha.exp() * next_logp.unsqueeze(2)),
                dim=1,
            ).unsqueeze(1)
            target_q_batch = reward_batch + self.train_param.GAMMA * (1 - done_batch) * value_prime_batch

        q_values1, _ = self.dp_q_net1(q_state_batch)
        q_values2, _ = self.dp_q_net2(q_state_batch)
        q1 = torch.gather(q_values1, 1, action_batch)
        q2 = torch.gather(q_values2, 1, action_batch)

        self.global_q_net1_optimizer.zero_grad()
        self.global_q_net2_optimizer.zero_grad()
        mse_loss = nn.MSELoss()
        q1_loss = mse_loss(q1, target_q_batch.detach()).mean()
        q2_loss = mse_loss(q2, target_q_batch.detach()).mean()
        q1_loss.backward()
        q2_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.global_q_net1.parameters(), max_norm=10, norm_type=2)
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.global_q_net2.parameters(), max_norm=10, norm_type=2)
        self.global_q_net1_optimizer.step()
        self.global_q_net2_optimizer.step()

        entropy = -(logp.exp() * logp).sum(dim=-1)
        alpha_loss = (self.log_alpha.exp() * (entropy.detach() - self.entropy_target)).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.target_q_update_counter += 1

        ave_reward = reward_batch.mean().item()
        if ave_reward > self.max_episodic_reward:
            self.max_episodic_reward = ave_reward
            self.models["curr_episode"] = runtime_state.curr_episode
            self.models["curr_samples"] = runtime_state.curr_samples
            self.models["log_alpha"] = self.log_alpha
            save_models(self.models, os.path.join(self.train_param.MODEL_PATH, "checkpoint_best.pth"))

        return {
            "reward_mean": reward_batch.mean().item(),
            "value_prime_mean": value_prime_batch.mean().item(),
            "policy_loss": policy_loss.item(),
            "q1_loss": q1_loss.item(),
            "entropy_mean": entropy.mean().item(),
            "policy_grad_norm": policy_grad_norm.item(),
            "q_grad_norm": q_grad_norm.item(),
            "log_alpha": self.log_alpha.item(),
            "alpha_loss": alpha_loss.item(),
        }

    def _train_sequence_batch(self, batch, runtime_state):
        state_batch = batch["state_batch"]
        next_state_batch = batch["next_state_batch"]
        q_state_batch = batch["q_state_batch"]
        next_q_state_batch = batch["next_q_state_batch"]
        action_batch = batch["action_batch"].long().squeeze(-1)
        reward_batch = batch["reward_batch"].squeeze(-1)
        done_batch = batch["done_batch"].squeeze(-1)
        valid_mask = batch["valid_mask"]

        policy_logp = self._forward_policy_sequence(state_batch, no_grad=False)
        with torch.no_grad():
            q_values1_policy, _ = self._forward_q_sequence(self.global_q_net1, q_state_batch, no_grad=True)
            q_values2_policy, _ = self._forward_q_sequence(self.global_q_net2, q_state_batch, no_grad=True)
            q_values_policy = torch.min(q_values1_policy, q_values2_policy)

        policy_terms = torch.sum(
            policy_logp.exp().unsqueeze(-1)
            * (self.log_alpha.exp().detach() * policy_logp.unsqueeze(-1) - q_values_policy.detach()),
            dim=2,
        )
        policy_loss = self._masked_mean(policy_terms, valid_mask)
        self.global_policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.global_policy_net.parameters(), max_norm=10, norm_type=2)
        self.global_policy_optimizer.step()

        with torch.no_grad():
            next_logp = self._forward_policy_sequence(next_state_batch, no_grad=True)
            next_q_values1, _ = self._forward_q_sequence(self.global_target_q_net1, next_q_state_batch, no_grad=True)
            next_q_values2, _ = self._forward_q_sequence(self.global_target_q_net2, next_q_state_batch, no_grad=True)
            next_q_values = torch.min(next_q_values1, next_q_values2)
            value_prime_batch = torch.sum(
                next_logp.unsqueeze(-1).exp() * (next_q_values - self.log_alpha.exp() * next_logp.unsqueeze(-1)),
                dim=2,
            )
            target_q_batch = reward_batch + self.train_param.GAMMA * (1 - done_batch) * value_prime_batch

        q_values1, _ = self._forward_q_sequence(self.global_q_net1, q_state_batch, no_grad=False)
        q_values2, _ = self._forward_q_sequence(self.global_q_net2, q_state_batch, no_grad=False)
        gather_indices = action_batch.unsqueeze(2)
        q1 = torch.gather(q_values1, 2, gather_indices).squeeze(2)
        q2 = torch.gather(q_values2, 2, gather_indices).squeeze(2)

        self.global_q_net1_optimizer.zero_grad()
        self.global_q_net2_optimizer.zero_grad()
        q1_loss = self._masked_mean((q1 - target_q_batch.detach()).pow(2), valid_mask)
        q2_loss = self._masked_mean((q2 - target_q_batch.detach()).pow(2), valid_mask)
        q1_loss.backward()
        q2_loss.backward()
        q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.global_q_net1.parameters(), max_norm=10, norm_type=2)
        q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.global_q_net2.parameters(), max_norm=10, norm_type=2)
        q_grad_norm = max(q1_grad_norm.item(), q2_grad_norm.item())
        self.global_q_net1_optimizer.step()
        self.global_q_net2_optimizer.step()

        entropy = -(policy_logp.exp() * policy_logp).sum(dim=-1, keepdim=True)
        alpha_loss = self._masked_mean(self.log_alpha.exp() * (entropy.detach() - self.entropy_target), valid_mask)
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.target_q_update_counter += 1

        ave_reward = self._masked_mean(reward_batch, valid_mask).item()
        if ave_reward > self.max_episodic_reward:
            self.max_episodic_reward = ave_reward
            self.models["curr_episode"] = runtime_state.curr_episode
            self.models["curr_samples"] = runtime_state.curr_samples
            self.models["log_alpha"] = self.log_alpha
            save_models(self.models, os.path.join(self.train_param.MODEL_PATH, "checkpoint_best.pth"))

        return {
            "reward_mean": self._masked_mean(reward_batch, valid_mask).item(),
            "value_prime_mean": self._masked_mean(value_prime_batch, valid_mask).item(),
            "policy_loss": policy_loss.item(),
            "q1_loss": q1_loss.item(),
            "entropy_mean": self._masked_mean(entropy, valid_mask).item(),
            "policy_grad_norm": policy_grad_norm.item(),
            "q_grad_norm": q_grad_norm,
            "log_alpha": self.log_alpha.item(),
            "alpha_loss": alpha_loss.item(),
        }

    def _maybe_update_target_networks(self):
        if self.target_q_update_counter <= self.train_param.TARGET_Q_NET_UPDATE_FREQ:
            return

        self.target_q_update_counter = 1
        if self.train_param.TARGET_Q_NET_UPDATE_SOFT:
            tau = 0.005
            for target_param, param in zip(self.global_target_q_net1.parameters(), self.global_q_net1.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.global_target_q_net2.parameters(), self.global_q_net2.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        else:
            self.global_target_q_net1.load_state_dict(self.global_q_net1.state_dict())
            self.global_target_q_net2.load_state_dict(self.global_q_net2.state_dict())

        self.global_target_q_net1.eval()
        self.global_target_q_net2.eval()

    def _collect_perf_data(self, runtime_state, metric_names):
        if not metric_names:
            return []

        perf_metrics = runtime_state.get_perf_metrics_snapshot()
        has_fresh_perf_metrics = any(len(perf_metrics[key]) > 0 for key in metric_names)
        if not has_fresh_perf_metrics:
            if self.last_perf_data is not None:
                return self.last_perf_data.copy()
            return [0.0 for _ in metric_names]

        perf_data = []
        for key in metric_names:
            if key == "spl":
                spl = calcu_spl(perf_metrics["success"], perf_metrics["travel_dist"])
                perf_data.append(spl)
            else:
                perf_data.append(np.nanmean(perf_metrics[key]))
        self.last_perf_data = perf_data.copy()
        return perf_data
