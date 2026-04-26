import math

import torch
import torch.nn as nn

import macronav.pretrain.models as pretrain_models

ENV_ENCODING_MODEL = "vit_tiny_patch8"


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, n_layers=3):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.functional.relu(x)
        return x


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, torch.finfo(U.dtype).min)
        attention = torch.log_softmax(U, dim=-1)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q
        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)
        V = torch.matmul(v_flat, self.w_value).view(shape_v)

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, torch.finfo(U.dtype).min)
        attention = torch.softmax(U, dim=-1)
        heads = torch.matmul(attention, V)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            self.w_out.view(-1, self.embedding_dim),
        ).view(-1, n_query, self.embedding_dim)
        return out, attention


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(
            nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, embedding_dim)
        )
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(
            nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, embedding_dim)
        )
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, attn_weights = self.multiHeadAttention(
            q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, attn_weights


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, out, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            out, attn_weights = layer(out, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return out, attn_weights


class PolicyNet(nn.Module):
    def __init__(self, configs: dict):
        input_dim = configs.get("input_dim", 7)
        embedding_dim = configs.get("embedding_dim", 128)
        self.k_size = configs.get("k_size", 20)
        self.use_local_nodes = True
        self.use_env_encoding = True
        self.use_lstm = False
        self.device = configs.get("device", "cpu")
        self.eval_mode = configs.get("eval_mode", False)
        model_args = configs.get(
            "policy_net_args",
            dict(
                encoder_layer=6,
                decoder_layer=1,
                encoder_head=8,
                decoder_head=8,
                lstm_layer=1,
                lstm_hidden_size=128,
                env_encoding_model=ENV_ENCODING_MODEL,
                env_encoding_model_use_pretrained=True,
                use_res_conn=True,
                env_encoding_freeze=False,
            ),
        )
        super(PolicyNet, self).__init__()

        self.initial_embedding = nn.Linear(input_dim, embedding_dim)  # layer for non-end position
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        self.use_res_conn = True

        self.nodes_encoder = Encoder(
            embedding_dim=embedding_dim, n_head=model_args["encoder_head"], n_layer=model_args["encoder_layer"]
        )
        self.nodes_decoder = Decoder(
            embedding_dim=embedding_dim, n_head=model_args["decoder_head"], n_layer=model_args["decoder_layer"]
        )
        self.pointer = SingleHeadAttention(embedding_dim)
        if self.use_res_conn:
            self.node_fuse_ln = nn.LayerNorm(embedding_dim)
            self.env_fuse_ln = nn.LayerNorm(embedding_dim)
        if self.use_lstm:
            self.curr_node_lstm = nn.LSTM(
                embedding_dim, model_args["lstm_hidden_size"], num_layers=model_args["lstm_layer"], batch_first=True
            )
            self.curr_node_lstm.flatten_parameters()
            self.lstm_ht = None  # [seq_len, batch, hidden_size]
            self.lstm_ct = None
        if self.use_env_encoding:
            if self.eval_mode:
                model_args["env_encoding_model_use_pretrained"] = False
            model_args["env_encoding_freeze"] = False
            self.explored_env_encoder = ExploredEnvEncoder(model_args).to(self.device)
            self.env_fuse_ln = nn.LayerNorm(embedding_dim)
        pass

    def reset_recurrent_state(self):
        if not self.use_lstm:
            return
        self.lstm_ht = None
        self.lstm_ct = None

    def set_recurrent_state(self, lstm_ht, lstm_ct):
        if not self.use_lstm:
            return
        self.lstm_ht = lstm_ht
        self.lstm_ct = lstm_ct

    def get_recurrent_state(self):
        if not self.use_lstm:
            return None, None
        return self.lstm_ht, self.lstm_ct

    def _run_policy_lstm(self, curr_node_feat):
        if self.lstm_ht is None:
            traj, (self.lstm_ht, self.lstm_ct) = self.curr_node_lstm(curr_node_feat)
        else:
            traj, (self.lstm_ht, self.lstm_ct) = self.curr_node_lstm(curr_node_feat, (self.lstm_ht, self.lstm_ct))
        return traj

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.nodes_encoder(
            src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask
        )
        return enhanced_node_feature

    def output_policy(
        self, encoded_node_feat, edge_inputs, current_index, edge_padding_mask, node_padding_mask, env_encoding=None
    ):
        curr_node_edges = edge_inputs.permute(0, 2, 1)
        embedding_dim = encoded_node_feat.size()[2]

        curr_node_feat = torch.gather(encoded_node_feat, 1, current_index.repeat(1, 1, embedding_dim))

        if edge_padding_mask is not None:
            curr_node_edge_mask = edge_padding_mask
        else:
            curr_node_edge_mask = None
        # curr_node_edge_mask[:, :, 0] = 1  # don't stay at current position

        if 1:
            # neighbor nodes decoding
            if not self.use_local_nodes:
                decoded_curr_node_feat, attention_weights = self.nodes_decoder(
                    curr_node_feat, encoded_node_feat, node_padding_mask
                )
                neigboring_feature = torch.gather(encoded_node_feat, 1, curr_node_edges.repeat(1, 1, embedding_dim))
            else:
                if self.use_lstm:
                    traj = self._run_policy_lstm(curr_node_feat)
                    decoded_traj_feat, attention_weights = self.nodes_decoder(
                        traj, encoded_node_feat, node_padding_mask
                    )
                    if self.use_res_conn:
                        decoded_curr_node_feat = decoded_traj_feat + curr_node_feat  # residual connection
                        decoded_curr_node_feat = self.node_fuse_ln(decoded_curr_node_feat)
                    else:
                        decoded_curr_node_feat = decoded_traj_feat
                else:
                    # fuse the neighboring feature into the current node feature
                    decoded_traj_feat, attention_weights = self.nodes_decoder(
                        curr_node_feat, encoded_node_feat, node_padding_mask
                    )
                    decoded_curr_node_feat = decoded_traj_feat + curr_node_feat
                neigboring_feature = encoded_node_feat[:, : self.k_size, :]
            if self.use_env_encoding:
                decoded_traj_feat, _ = self.explored_env_encoder.node_env_fuser(decoded_curr_node_feat, env_encoding)
                if self.use_res_conn:
                    decoded_curr_node_feat = decoded_curr_node_feat + decoded_traj_feat  # residual connection
                    decoded_curr_node_feat = self.env_fuse_ln(decoded_curr_node_feat)
                else:
                    decoded_curr_node_feat = decoded_traj_feat
        else:  # first fuse neighor nodes and env encoding, then decode policy
            if self.use_lstm:
                traj = self._run_policy_lstm(curr_node_feat)
            if self.use_env_encoding:
                fused_node_feat, _ = self.explored_env_encoder.node_env_fuser(encoded_node_feat, env_encoding)
                encoded_node_feat = encoded_node_feat + fused_node_feat

            decoded_traj_feat_, attention_weights = self.nodes_decoder(traj, encoded_node_feat, node_padding_mask)
            traj = traj + decoded_traj_feat_
            decoded_curr_node_feat = traj
            neigboring_feature = encoded_node_feat[:, : self.k_size, :]

        decoded_curr_node_feat = self.current_embedding(
            torch.cat((decoded_curr_node_feat, curr_node_feat), dim=-1)
        )  # prevent catastrophic forgetting
        logp = self.pointer(decoded_curr_node_feat, neigboring_feature, curr_node_edge_mask)
        logp = logp.squeeze(1)

        return logp

    def forward(self, x):
        if self.use_lstm:
            self.curr_node_lstm.flatten_parameters()
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, gridmap_inputs = x
        env_encoding = None
        if self.use_env_encoding:
            if gridmap_inputs is None:
                raise ValueError("Env encoding is enabled but no gridmap input is provided")
            env_encoding = self.explored_env_encoder(gridmap_inputs)
        encoded_node_feat = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        logp = self.output_policy(
            encoded_node_feat, edge_inputs, current_index, edge_padding_mask, node_padding_mask, env_encoding
        )
        return logp


class QNet(nn.Module):
    def __init__(self, configs: dict):
        super(QNet, self).__init__()
        input_dim = configs.get("input_dim", 7)
        embedding_dim = configs.get("embedding_dim", 128)
        k_size = configs.get("k_size", 20)
        use_local_nodes = True
        use_env_encoding = False
        use_lstm = False
        model_args = configs.get(
            "q_net_args",
            dict(
                encoder_layer=6,
                decoder_layer=1,
                encoder_head=8,
                decoder_head=8,
                lstm_layer=1,
                lstm_hidden_size=128,
                env_encoding_model=ENV_ENCODING_MODEL,
                use_res_conn=True,
            ),
        )
        self.k_size = k_size
        self.initial_embedding = nn.Linear(input_dim, embedding_dim)  # layer for non-end position
        self.action_embedding = nn.Linear(embedding_dim * 3, embedding_dim)
        self.use_local_nodes = use_local_nodes
        self.use_env_encoding = use_env_encoding
        self.use_lstm = use_lstm
        self.use_res_conn = True

        self.nodes_encoder = Encoder(
            embedding_dim=embedding_dim, n_head=model_args["encoder_head"], n_layer=model_args["encoder_layer"]
        )
        self.nodes_decoder = Decoder(
            embedding_dim=embedding_dim, n_head=model_args["decoder_head"], n_layer=model_args["decoder_layer"]
        )
        self.q_values_layer = nn.Linear(embedding_dim, 1)
        if self.use_res_conn:
            self.node_fuse_ln = nn.LayerNorm(embedding_dim)
            self.env_fuse_ln = nn.LayerNorm(embedding_dim)

        if self.use_lstm:
            self.curr_node_lstm = nn.LSTM(
                embedding_dim, model_args["lstm_hidden_size"], num_layers=model_args["lstm_layer"], batch_first=True
            )
            self.lstm_ht = None  # [seq_len, batch, hidden_size]
            self.lstm_ct = None
        if self.use_env_encoding:
            self.explored_env_encoder = ExploredEnvEncoder(model_args)
            if self.use_res_conn:
                self.env_fuse_ln = nn.LayerNorm(embedding_dim)

    def reset_recurrent_state(self):
        if not self.use_lstm:
            return
        self.lstm_ht = None
        self.lstm_ct = None

    def _run_q_lstm(self, curr_node_feat):
        if self.lstm_ht is None:
            traj, (self.lstm_ht, self.lstm_ct) = self.curr_node_lstm(curr_node_feat)
        else:
            traj, (self.lstm_ht, self.lstm_ct) = self.curr_node_lstm(curr_node_feat, (self.lstm_ht, self.lstm_ct))
        return traj

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        embedding_feature = self.initial_embedding(node_inputs)
        embedding_feature = self.nodes_encoder(
            src=embedding_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask
        )

        return embedding_feature

    def output_q_values(
        self, encoded_node_feat, edge_inputs, current_index, edge_padding_mask, node_padding_mask, env_encoding=None
    ):
        k_size = edge_inputs.size()[2]
        current_edge = edge_inputs
        current_edge = current_edge.permute(0, 2, 1)
        embedding_dim = encoded_node_feat.size()[2]

        curr_node_feat = torch.gather(encoded_node_feat, 1, current_index.repeat(1, 1, embedding_dim))

        if not self.use_local_nodes:
            decoded_curr_node_feat, attention_weights = self.nodes_decoder(
                curr_node_feat, encoded_node_feat, node_padding_mask
            )
            neigboring_feature = torch.gather(encoded_node_feat, 1, current_edge.repeat(1, 1, embedding_dim))
        else:
            if self.use_lstm:
                traj = self._run_q_lstm(curr_node_feat)
                decoded_curr_node_feat_, attention_weights = self.nodes_decoder(
                    traj, encoded_node_feat, node_padding_mask
                )
                decoded_curr_node_feat = decoded_curr_node_feat_ + curr_node_feat
            else:
                decoded_curr_node_feat_, attention_weights = self.nodes_decoder(
                    curr_node_feat, encoded_node_feat, node_padding_mask
                )
                if self.use_res_conn:
                    decoded_curr_node_feat = decoded_curr_node_feat_ + curr_node_feat
                    decoded_curr_node_feat = self.node_fuse_ln(decoded_curr_node_feat)
                else:
                    decoded_curr_node_feat = decoded_curr_node_feat_
            neigboring_feature = encoded_node_feat[:, : self.k_size, :]

        if self.use_env_encoding:
            decoded_curr_node_feat_, _ = self.explored_env_encoder.node_env_fuser(decoded_curr_node_feat, env_encoding)
            if self.use_res_conn:
                decoded_curr_node_feat = decoded_curr_node_feat + decoded_curr_node_feat_
                decoded_curr_node_feat = self.env_fuse_ln(decoded_curr_node_feat)
            else:
                decoded_curr_node_feat = decoded_curr_node_feat_

        action_features = torch.cat(
            (
                decoded_curr_node_feat.repeat(1, k_size, 1),
                curr_node_feat.repeat(1, k_size, 1),
                neigboring_feature,
            ),
            dim=-1,
        )
        action_features = self.action_embedding(action_features)
        q_values = self.q_values_layer(action_features)

        if edge_padding_mask is not None:
            current_mask = edge_padding_mask
        else:
            current_mask = None
        current_mask[:, :, 0] = 1  # don't stay at current position
        current_mask = current_mask.permute(0, 2, 1)
        zero = torch.zeros_like(q_values).to(q_values.device)
        q_values = torch.where(current_mask == 1, zero, q_values)
        return q_values, attention_weights

    def forward(self, x):
        if self.use_lstm:
            self.curr_node_lstm.flatten_parameters()
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, gridmap_inputs = x
        env_encoding = None
        if gridmap_inputs is not None and self.use_env_encoding:
            env_encoding = self.explored_env_encoder(gridmap_inputs)
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        q_values, attention_weights = self.output_q_values(
            enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask, env_encoding
        )
        return q_values, attention_weights


class ExploredEnvEncoder(nn.Module):
    def __init__(self, model_args):
        super(ExploredEnvEncoder, self).__init__()
        self.env_encoding_model = model_args.get("env_encoding_model", ENV_ENCODING_MODEL)
        self.env_encoding_model_use_pretrained = model_args["env_encoding_model_use_pretrained"]
        embedding_dim = model_args["embedding_dim"] if "embedding_dim" in model_args else 128
        if self.env_encoding_model != ENV_ENCODING_MODEL:
            raise ValueError(
                f"Only release env encoding model '{ENV_ENCODING_MODEL}' is supported, got: {self.env_encoding_model}"
            )

        self.encoder = pretrain_models.__dict__[self.env_encoding_model]()
        if self.env_encoding_model_use_pretrained:
            if "env_encoding_model_ckpt" not in model_args:
                raise ValueError("Please provide the path to the pretrained model")
            self.encoder.load_ssl_weights(model_args["env_encoding_model_ckpt"], strict=True)

        self.encoder_out_lin = nn.Linear(self.encoder.embed_pred_head.out_features, embedding_dim)
        self.node_env_fuser = Decoder(
            embedding_dim=embedding_dim, n_head=model_args["decoder_head"], n_layer=model_args["decoder_layer"]
        )

    def freeze_encoder(self):
        """Freeze the weights of the environment encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("ExploredEnvEncoder weights frozen")

    def unfreeze_encoder(self):
        """Unfreeze the weights of the environment encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("ExploredEnvEncoder weights unfrozen")

    def forward(self, x):
        """
        x: gridmap image tensor, shape: (batch_size, 3, H, W)
        """
        x_ = self.encoder(x)
        x = self.encoder_out_lin(x_)
        return x
