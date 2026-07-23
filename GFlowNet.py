import torch
import torch.nn as nn
import numpy as np


class ForwardPolicy(nn.Module):
	"""Forward transition P_F(s_{t+1} | s_t; theta)"""

	def __init__(self, embed_dim, block_dim, context_dim, hidden_dim):
		super(ForwardPolicy, self).__init__()
		input_dim = embed_dim + context_dim
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.LeakyReLU(0.2),
		)
		self.mean_head = nn.Linear(hidden_dim, block_dim)
		self.log_std_head = nn.Linear(hidden_dim, block_dim)

	def forward(self, state, context):
		h = torch.cat([state, context], dim=-1)
		h = self.net(h)
		mean = self.mean_head(h)
		log_std = self.log_std_head(h).clamp(-5.0, 2.0)
		return mean, log_std


class BackwardPolicy(nn.Module):
	"""Backward transition P_B(s_t | s_{t+1}; theta)"""

	def __init__(self, embed_dim, block_dim, context_dim, hidden_dim):
		super(BackwardPolicy, self).__init__()
		input_dim = embed_dim + context_dim
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.LeakyReLU(0.2),
		)
		self.mean_head = nn.Linear(hidden_dim, block_dim)
		self.log_std_head = nn.Linear(hidden_dim, block_dim)

	def forward(self, state, context):
		h = torch.cat([state, context], dim=-1)
		h = self.net(h)
		mean = self.mean_head(h)
		log_std = self.log_std_head(h).clamp(-5.0, 2.0)
		return mean, log_std


class GFlowNet(nn.Module):
	"""
	Generative Flow Network for reward-proportional embedding generation.

	Generates user embeddings x0 in R^d by sequentially generating K blocks,
	where the probability of generating x0 is proportional to reward R(x0).

	Trained with Trajectory Balance (TB) loss:
		L_TB = (log Z + sum(log P_F) - log R(x0) - sum(log P_B))^2

	Reward:
		R(x0) = exp((Score_BPR(x0) + beta * NDCG@K(x0)) / tau)
	"""

	def __init__(self, embed_dim, num_blocks, context_dim, hidden_dim=256,
				 reward_temp=1.0, beta_ndcg=0.0, topk=20):
		super(GFlowNet, self).__init__()
		assert embed_dim % num_blocks == 0, \
			f"embed_dim ({embed_dim}) must be divisible by num_blocks ({num_blocks})"

		self.embed_dim = embed_dim
		self.num_blocks = num_blocks
		self.block_dim = embed_dim // num_blocks
		self.reward_temp = reward_temp
		self.beta_ndcg = beta_ndcg
		self.topk = topk

		# Learnable log partition function Z
		self.log_Z = nn.Parameter(torch.zeros(1))

		# Step embedding so policies know which block is being generated
		self.step_embed = nn.Embedding(num_blocks, hidden_dim)

		# Augmented context includes step embedding
		aug_context_dim = context_dim + hidden_dim
		self.forward_policy = ForwardPolicy(embed_dim, self.block_dim, aug_context_dim, hidden_dim)
		self.backward_policy = BackwardPolicy(embed_dim, self.block_dim, aug_context_dim, hidden_dim)

		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				size = m.weight.size()
				std = np.sqrt(2.0 / (size[0] + size[1]))
				m.weight.data.normal_(0.0, std)
				if m.bias is not None:
					m.bias.data.normal_(0.0, 0.001)

	def _aug_context(self, context, step, batch_size):
		"""Concatenate context with step embedding."""
		step_emb = self.step_embed.weight[step].unsqueeze(0).expand(batch_size, -1)
		return torch.cat([context, step_emb], dim=-1)

	def _log_gaussian(self, x, mean, log_std):
		"""Log probability under diagonal Gaussian N(mean, diag(exp(log_std)^2))."""
		var = (2.0 * log_std).exp()
		return -0.5 * (
			self.block_dim * np.log(2.0 * np.pi)
			+ 2.0 * log_std.sum(dim=-1)
			+ ((x - mean).pow(2) / var.clamp(min=1e-8)).sum(dim=-1)
		)

	def sample_trajectory(self, context):
		"""
		Sample trajectory s_0 -> s_1 -> ... -> s_K = x_0.

		Args:
			context: (batch, context_dim) user context features

		Returns:
			x0: (batch, embed_dim) generated embedding
			log_pf_sum: (batch,) sum of log forward transition probs
			log_pb_sum: (batch,) sum of log backward transition probs
		"""
		batch_size = context.shape[0]
		device = context.device

		state = torch.zeros(batch_size, self.embed_dim, device=device)
		states = [state]
		blocks = []
		log_pf_sum = torch.zeros(batch_size, device=device)

		# Forward pass: generate blocks sequentially
		for t in range(self.num_blocks):
			ctx = self._aug_context(context, t, batch_size)
			mean_f, log_std_f = self.forward_policy(state, ctx)

			# Sample block from forward policy
			block = mean_f + log_std_f.exp() * torch.randn_like(mean_f)
			blocks.append(block)

			# Accumulate log P_F(s_{t+1} | s_t)
			log_pf = self._log_gaussian(block, mean_f, log_std_f)
			log_pf_sum = log_pf_sum + log_pf

			# Update state: fill in block t
			state = state.clone()
			s, e = t * self.block_dim, (t + 1) * self.block_dim
			state[:, s:e] = block
			states.append(state)

		x0 = state

		# Backward pass: compute log P_B for each transition
		log_pb_sum = torch.zeros(batch_size, device=device)
		for t in range(self.num_blocks):
			ctx = self._aug_context(context, t, batch_size)
			s_next = states[t + 1]

			mean_b, log_std_b = self.backward_policy(s_next, ctx)
			log_pb = self._log_gaussian(blocks[t], mean_b, log_std_b)
			log_pb_sum = log_pb_sum + log_pb

		return x0, log_pf_sum, log_pb_sum

	def compute_reward(self, x0, modal_feats, interactions):
		"""
		Compute log reward:
			log R(x0) = (Score_BPR(x0) + beta * NDCG@K(x0)) / tau

		Args:
			x0: (batch, embed_dim) generated user embeddings
			modal_feats: (num_items, embed_dim) item features for this modality
			interactions: (batch, num_items) binary ground truth interactions

		Returns:
			log_reward: (batch,)
		"""
		# Predicted interaction scores: x0 @ modal_feats^T
		scores = torch.mm(x0, modal_feats.t())

		# --- Score_BPR: mean positive score - mean negative score ---
		pos_mask = (interactions > 0).float()
		neg_mask = 1.0 - pos_mask
		n_pos = pos_mask.sum(dim=-1).clamp(min=1.0)
		n_neg = neg_mask.sum(dim=-1).clamp(min=1.0)

		bpr_score = (scores * pos_mask).sum(-1) / n_pos - (scores * neg_mask).sum(-1) / n_neg

		# --- NDCG@K ---
		ndcg = self._compute_ndcg(scores, interactions, self.topk)

		# log R(x0) = (Score_BPR + beta * NDCG@K) / tau
		log_reward = (bpr_score + self.beta_ndcg * ndcg) / self.reward_temp
		return log_reward

	def _compute_ndcg(self, scores, interactions, k):
		"""Compute NDCG@K per user (non-differentiable, fine for GFlowNet reward)."""
		actual_k = min(k, scores.shape[1])

		_, topk_idx = torch.topk(scores, k=actual_k, dim=-1)
		rel = torch.gather(interactions, 1, topk_idx)

		positions = torch.arange(1, actual_k + 1, dtype=torch.float32, device=scores.device)
		discounts = 1.0 / torch.log2(positions + 1.0)

		dcg = (rel * discounts.unsqueeze(0)).sum(dim=-1)

		ideal_rel, _ = interactions.sort(dim=-1, descending=True)
		ideal_rel = ideal_rel[:, :actual_k]
		idcg = (ideal_rel * discounts.unsqueeze(0)).sum(dim=-1)

		return dcg / idcg.clamp(min=1e-8)

	def compute_tb_loss(self, log_pf_sum, log_pb_sum, log_reward):
		"""
		Trajectory Balance Loss:
			L_TB = (log Z + sum(log P_F) - log R(x0) - sum(log P_B))^2
		"""
		loss = (self.log_Z + log_pf_sum - log_reward - log_pb_sum).pow(2)
		return loss.mean()

	def training_losses(self, context, modal_feats, interactions):
		"""
		Full training step: sample trajectory, compute reward, return TB loss.

		Args:
			context: (batch, embed_dim) user context
			modal_feats: (num_items, embed_dim) modal item features
			interactions: (batch, num_items) ground truth interactions

		Returns:
			tb_loss: scalar TB loss
			log_reward_mean: scalar mean log reward (for monitoring)
		"""
		x0, log_pf_sum, log_pb_sum = self.sample_trajectory(context)

		with torch.no_grad():
			log_reward = self.compute_reward(x0, modal_feats, interactions)

		tb_loss = self.compute_tb_loss(log_pf_sum, log_pb_sum, log_reward)
		return tb_loss, log_reward.mean()

	@torch.no_grad()
	def sample(self, context):
		"""
		Inference: generate x0 using forward policy means (deterministic).

		Args:
			context: (batch, embed_dim) user context

		Returns:
			x0: (batch, embed_dim) generated embedding
		"""
		batch_size = context.shape[0]
		device = context.device
		state = torch.zeros(batch_size, self.embed_dim, device=device)

		for t in range(self.num_blocks):
			ctx = self._aug_context(context, t, batch_size)
			mean_f, _ = self.forward_policy(state, ctx)

			state = state.clone()
			s, e = t * self.block_dim, (t + 1) * self.block_dim
			state[:, s:e] = mean_f

		return state
