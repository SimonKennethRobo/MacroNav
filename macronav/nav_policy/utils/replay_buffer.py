import bisect
import numpy as np
import torch


class PrefixSum:
    def __init__(self, max_len):
        self.ar = []
        self.max_len = max_len
        self.prefix_sum = np.zeros(1, dtype=np.int64)
        self.curr_max = 0

    def _rebuild_prefix_sum(self):
        self.prefix_sum = np.zeros(len(self.ar) + 1, dtype=np.int64)
        if self.ar:
            self.prefix_sum[1:] = np.cumsum(self.ar, dtype=np.int64)
        self.curr_max = int(self.prefix_sum[-1])

    def add(self, val):
        self.ar.append(int(val))
        if len(self.ar) > self.max_len:
            self.ar = self.ar[-self.max_len :]
        self._rebuild_prefix_sum()

    def pop_left(self):
        if not self.ar:
            return 0
        removed = self.ar.pop(0)
        self._rebuild_prefix_sum()
        return removed

    def get_range_idx(self, idx):
        """get the range index of the idx-th element"""
        if idx > self.curr_max - 1:
            raise ValueError("Index out of range.")
        return bisect.bisect_right(self.prefix_sum, idx) - 1

    def get_range_relative_idx(self, idx, range_idx):
        """relative index in a range"""
        return idx - self.prefix_sum[range_idx]

    def get_range_start(self, range_idx):
        return int(self.prefix_sum[range_idx])

    def __len__(self):
        return len(self.ar)


def _sample_transition_batch(replay_buffer, train_param: dict, device):
    buffer_size = len(replay_buffer["done"])
    indices = range(min(buffer_size, train_param["replay_buffer_size"]))

    sample_indices = np.random.choice(indices, train_param["batch_size"], replace=False)

    gridmap_batch = torch.stack([replay_buffer["gridmap_inputs"][index] for index in sample_indices]).to(device)
    next_gridmap_batch = torch.stack([replay_buffer["next_gridmap_inputs"][index] for index in sample_indices]).to(
        device
    )

    rollouts = {}
    for key in train_param["replay_buffer_keys"]:
        if key == "gridmap_inputs":
            rollouts["gridmap_inputs"] = gridmap_batch
        elif key == "next_gridmap_inputs":
            rollouts["next_gridmap_inputs"] = next_gridmap_batch
        else:
            rollouts[key] = [replay_buffer[key][index] for index in sample_indices]

    node_inputs_batch = torch.stack(rollouts["node_inputs"]).to(device)
    edge_inputs_batch = torch.stack(rollouts["edge_inputs"]).to(device)
    current_index_batch = torch.stack(rollouts["current_index"]).to(device)
    node_padding_mask_batch = torch.stack(rollouts["node_padding_mask"]).to(device)
    edge_padding_mask_batch = torch.stack(rollouts["curr_node_edge_padding_mask"]).to(device)
    edge_mask_batch = torch.stack(rollouts["edge_mask"]).to(device)
    q_edge_mask_batch = (
        torch.stack(rollouts["q_edge_mask"]).to(device) if "q_edge_mask" in rollouts else edge_mask_batch
    )
    action_batch = torch.stack(rollouts["action"]).to(device)
    reward_batch = torch.stack(rollouts["reward"]).to(device)
    done_batch = torch.stack(rollouts["done"]).to(device)
    next_node_inputs_batch = torch.stack(rollouts["next_node_inputs"]).to(device)
    next_edge_inputs_batch = torch.stack(rollouts["next_edge_inputs"]).to(device)
    next_current_index_batch = torch.stack(rollouts["next_current_index"]).to(device)
    next_node_padding_mask_batch = torch.stack(rollouts["next_node_padding_mask"]).to(device)
    next_edge_padding_mask_batch = torch.stack(rollouts["next_curr_node_edge_padding_mask"]).to(device)
    next_edge_mask_batch = torch.stack(rollouts["next_edge_mask"]).to(device)
    next_q_edge_mask_batch = (
        torch.stack(rollouts["next_q_edge_mask"]).to(device)
        if "next_q_edge_mask" in rollouts
        else next_edge_mask_batch
    )

    model_input = (
        node_inputs_batch,
        edge_inputs_batch,
        current_index_batch,
        node_padding_mask_batch,
        edge_padding_mask_batch,
        edge_mask_batch,
        gridmap_batch,
    )
    model_input_next = (
        next_node_inputs_batch,
        next_edge_inputs_batch,
        next_current_index_batch,
        next_node_padding_mask_batch,
        next_edge_padding_mask_batch,
        next_edge_mask_batch,
        next_gridmap_batch,
    )
    q_model_input = (
        node_inputs_batch,
        edge_inputs_batch,
        current_index_batch,
        node_padding_mask_batch,
        edge_padding_mask_batch,
        q_edge_mask_batch,
        gridmap_batch,
    )
    q_model_input_next = (
        next_node_inputs_batch,
        next_edge_inputs_batch,
        next_current_index_batch,
        next_node_padding_mask_batch,
        next_edge_padding_mask_batch,
        next_q_edge_mask_batch,
        next_gridmap_batch,
    )
    return {
        "is_sequence": False,
        "state_batch": model_input,
        "next_state_batch": model_input_next,
        "q_state_batch": q_model_input,
        "next_q_state_batch": q_model_input_next,
        "action_batch": action_batch,
        "reward_batch": reward_batch,
        "done_batch": done_batch,
        "valid_mask": None,
    }


def _build_zero_padding(reference_tensor):
    return torch.zeros_like(reference_tensor)


def _sample_sequence_batch(replay_buffer, train_param: dict, device, episode_lens_prefix_sum: PrefixSum):
    if episode_lens_prefix_sum is None or len(episode_lens_prefix_sum) == 0:
        raise ValueError("Sequence sampling requires a non-empty episode_lens_prefix_sum")

    sequence_length = int(train_param.get("replay_sequence_length", 1))
    if sequence_length <= 0:
        raise ValueError(f"replay_sequence_length must be positive, got {sequence_length}")

    batch_size = int(train_param["batch_size"])
    total_steps = len(replay_buffer["done"])
    if total_steps == 0:
        raise ValueError("Replay buffer is empty")

    sample_keys = list(train_param["replay_buffer_keys"])
    sampled = {key: [] for key in sample_keys}
    valid_masks = []
    zero_refs = {key: _build_zero_padding(replay_buffer[key][0]) for key in sample_keys}

    for _ in range(batch_size):
        global_idx = int(np.random.randint(0, episode_lens_prefix_sum.curr_max))
        episode_idx = episode_lens_prefix_sum.get_range_idx(global_idx)
        episode_start = episode_lens_prefix_sum.get_range_start(episode_idx)
        episode_len = episode_lens_prefix_sum.ar[episode_idx]
        local_start = episode_lens_prefix_sum.get_range_relative_idx(global_idx, episode_idx)
        actual_len = min(sequence_length, episode_len - local_start)

        sequence_indices = range(episode_start + local_start, episode_start + local_start + actual_len)
        for key in sample_keys:
            sequence_items = [replay_buffer[key][idx] for idx in sequence_indices]
            if actual_len < sequence_length:
                sequence_items.extend(
                    [_build_zero_padding(zero_refs[key]) for _ in range(sequence_length - actual_len)]
                )
            sampled[key].append(torch.stack(sequence_items))

        valid_mask = torch.zeros(sequence_length, 1, dtype=torch.float32)
        valid_mask[:actual_len] = 1.0
        valid_masks.append(valid_mask)

    rollouts = {key: torch.stack(sampled[key]).to(device) for key in sample_keys}
    valid_mask = torch.stack(valid_masks).to(device)

    q_edge_mask_batch = rollouts["q_edge_mask"] if "q_edge_mask" in rollouts else rollouts["edge_mask"]
    next_q_edge_mask_batch = (
        rollouts["next_q_edge_mask"] if "next_q_edge_mask" in rollouts else rollouts["next_edge_mask"]
    )

    model_input = (
        rollouts["node_inputs"],
        rollouts["edge_inputs"],
        rollouts["current_index"],
        rollouts["node_padding_mask"],
        rollouts["curr_node_edge_padding_mask"],
        rollouts["edge_mask"],
        rollouts.get("gridmap_inputs"),
    )
    model_input_next = (
        rollouts["next_node_inputs"],
        rollouts["next_edge_inputs"],
        rollouts["next_current_index"],
        rollouts["next_node_padding_mask"],
        rollouts["next_curr_node_edge_padding_mask"],
        rollouts["next_edge_mask"],
        rollouts.get("next_gridmap_inputs"),
    )
    q_model_input = (
        rollouts["node_inputs"],
        rollouts["edge_inputs"],
        rollouts["current_index"],
        rollouts["node_padding_mask"],
        rollouts["curr_node_edge_padding_mask"],
        q_edge_mask_batch,
        rollouts.get("gridmap_inputs"),
    )
    q_model_input_next = (
        rollouts["next_node_inputs"],
        rollouts["next_edge_inputs"],
        rollouts["next_current_index"],
        rollouts["next_node_padding_mask"],
        rollouts["next_curr_node_edge_padding_mask"],
        next_q_edge_mask_batch,
        rollouts.get("next_gridmap_inputs"),
    )
    return {
        "is_sequence": True,
        "state_batch": model_input,
        "next_state_batch": model_input_next,
        "q_state_batch": q_model_input,
        "next_q_state_batch": q_model_input_next,
        "action_batch": rollouts["action"],
        "reward_batch": rollouts["reward"],
        "done_batch": rollouts["done"],
        "valid_mask": valid_mask,
    }


def sample_batch(replay_buffer, train_param: dict, device, episode_lens_prefix_sum: PrefixSum = None):
    recurrent_enabled = False
    use_sequence_sample = False
    sequence_length = int(train_param.get("replay_sequence_length", 1))
    if recurrent_enabled and use_sequence_sample and sequence_length > 1:
        return _sample_sequence_batch(replay_buffer, train_param, device, episode_lens_prefix_sum)
    return _sample_transition_batch(replay_buffer, train_param, device)


class DictReplayBuffer:
    def __init__(self, max_size, keys, device="cpu", logger=None, img_compressed=False):
        self.max_size = max_size
        self.buffer = {key: [] for key in keys}
        self.device = device
        self.img_compressed = img_compressed
        if logger != None:
            self.logprint = logger.info
        else:
            self.logprint = print
        self.episode_lens_prefix_sum = PrefixSum(max_size)

    def add(self, episode_data):
        for key in episode_data.keys():
            self.buffer[key].extend(episode_data[key])
        self.episode_lens_prefix_sum.add(len(episode_data["done"]))
        while len(self.buffer["done"]) > self.max_size and len(self.episode_lens_prefix_sum) > 0:
            self.logprint("Replay buffer overflow")
            trim_steps = self.episode_lens_prefix_sum.pop_left()
            for key in self.buffer.keys():
                self.buffer[key] = self.buffer[key][trim_steps:]

    def sample(self, batch_size):
        raise NotImplementedError("DictReplayBuffer.sample is unused; call sample_batch with train config instead.")

    def __len__(self):
        return len(self.buffer["done"])

if __name__ == "__main__":
    if 0:  # test prefix sum
        prefix_sum = PrefixSum(5)
        prefix_sum.add(2)
        print(f"ar: {prefix_sum.ar}")
        print(f"prefix_sum: {prefix_sum.prefix_sum}")
        range_idx = prefix_sum.get_range_idx(0)
        print(0, range_idx, prefix_sum.get_range_relative_idx(0, range_idx))

        prefix_sum.add(2)
        print(f"ar: {prefix_sum.ar}")
        print(f"prefix_sum: {prefix_sum.prefix_sum}")
        range_idx = prefix_sum.get_range_idx(1)
        print(1, range_idx, prefix_sum.get_range_relative_idx(1, range_idx))

        prefix_sum.add(3)
        prefix_sum.add(2)
        prefix_sum.add(1)
        print(f"ar: {prefix_sum.ar}")
        print(f"prefix_sum: {prefix_sum.prefix_sum}")
        for i in range(4, 9):
            range_idx = prefix_sum.get_range_idx(i)
            print(i, range_idx, prefix_sum.get_range_relative_idx(i, range_idx))

        prefix_sum.add(2)
        prefix_sum.add(2)
        print(f"new ar: {prefix_sum.ar}")
        print(f"new prefix_sum: {prefix_sum.prefix_sum}")
        for i in range(4, 9):
            range_idx = prefix_sum.get_range_idx(i)
            print(i, range_idx, prefix_sum.get_range_relative_idx(i, range_idx))
