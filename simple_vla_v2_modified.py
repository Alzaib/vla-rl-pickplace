import torch
import torch.nn as nn


class SimpleVLAv2(nn.Module):
    """
    Same idea as SimpleVLA (image tokens + text tokens + time token + ACT token -> transformer ->
    3 action heads), with the changes that matter for this task:

    1. CELL-ALIGNED PATCH TOKENS.  The image is 72x84 px and one grid cell is 6x6 px.  A 6x6
       patch embedding gives a 12x14 grid of tokens where each token is exactly one cell
       (v1 had 5x6 tokens, each covering ~2.4 x 2.3 cells, with blocks only 0.67 cell wide).
       Actions are +-1 cell, so one-cell precision is what the policy needs.
    2. FACTORISED ROW + COLUMN position embeddings.  The position of token (r, c) is
       row_emb[r] + col_emb[c], so "how many cells left/right/up/down is A from B" is
       an easy (nearly linear) function of the embeddings.
    3. PRE-NORM transformer with a final LayerNorm (more stable to train from scratch).
    4. AUXILIARY HEAD (optional use): from the ACT token, predict where the end-effector, the
       target block and the target region are.  Extra supervision that pushes the network to
       actually find the objects the instruction talks about.  Not used at test time.
    """

    def __init__(self, vocab_size, d_model=128, max_text_len=8, n_heads=4, n_layers=4,
                 patch=6, grid_hw=(12, 14), n_aux=6, aux_classes=12, dropout=0.0):
        super().__init__()
        self.patch, self.grid_hw = patch, grid_hw
        H, W = grid_hw

        # ---- vision: one token per grid cell
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)
        self.patch_mlp = nn.Sequential(nn.GELU(), nn.Linear(d_model, d_model))
        self.patch_norm = nn.LayerNorm(d_model)
        self.row_emb = nn.Parameter(torch.randn(H, d_model) * 0.5)
        self.col_emb = nn.Parameter(torch.randn(W, d_model) * 0.5)

        # ---- language
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.text_pos_emb = nn.Parameter(torch.randn(max_text_len, d_model) * 0.5)

        # ---- time + ACT token
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, d_model))
        self.act_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.5)

        # ---- transformer
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
                                                 enable_nested_tensor=False)

        # ---- heads
        self.dx_head = nn.Linear(d_model, 3)
        self.dy_head = nn.Linear(d_model, 3)
        self.grip_head = nn.Linear(d_model, 2)
        self.n_aux, self.aux_classes = n_aux, aux_classes
        self.aux_head = nn.Linear(d_model, n_aux * aux_classes)

    def encode(self, image, text_tokens, time_step):
        """image (B,3,72,84) float, text_tokens (B,8) long, time_step (B,) -> ACT-token features (B, d_model)"""
        B = image.shape[0]
        H, W = self.grid_hw
        assert image.shape[-2:] == (H * self.patch, W * self.patch), f"unexpected image size {tuple(image.shape)}"

        # image tokens: B x (H*W) x d, row-major (token index = row * W + col)
        x = self.patch_embed(image).flatten(2).transpose(1, 2)
        x = self.patch_norm(self.patch_mlp(x))
        pos = (self.row_emb[:, None, :] + self.col_emb[None, :, :]).reshape(H * W, -1)
        image_tokens = x + pos.unsqueeze(0)

        text = self.word_emb(text_tokens) + self.text_pos_emb.unsqueeze(0)
        time_token = self.time_mlp(time_step.float().unsqueeze(-1)).unsqueeze(1)
        act = self.act_token.expand(B, -1, -1)

        tokens = torch.cat([act, image_tokens, text, time_token], dim=1)     # 1 + 168 + 8 + 1 = 178 tokens
        return self.transformer(tokens)[:, 0]

    def forward(self, image, text_tokens, time_step):
        act_out = self.encode(image, text_tokens, time_step)
        B = act_out.shape[0]
        return {
            "dx": self.dx_head(act_out),
            "dy": self.dy_head(act_out),
            "grip": self.grip_head(act_out),
            "aux": self.aux_head(act_out).view(B, self.n_aux, self.aux_classes),
        }
