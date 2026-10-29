import torch
import torch.nn as nn

PAD_DX, PAD_DY, PAD_GRIP = 3, 3, 2        # "no action yet" class ids (see chunk_data.PAD)


class SimpleVLAChunk(nn.Module):
    """
    SimpleVLAv2 turned into an action-chunking policy.

        observation_t (image, text, time) + last H performed actions
                    -->  [a_t, a_t+1, ..., a_t+K-1]

    Token sequence fed to the transformer (all width d_model):
        K  chunk-query tokens   (learned; query k will become the prediction for step t+k)
        168 image tokens        (6x6 patch = one grid cell each, row+col position embeddings)
        8  text tokens
        1  time token
        H  history tokens       (embedding of each past action + "how long ago" embedding)
    The K query outputs go through shared heads -> logits of shape (B, K, 3) for dx, (B, K, 3) dy, (B, K, 2) grip.
    K=1, H=0 is (up to the extra query-position embedding) the same model as SimpleVLAv2, so use it as the baseline.

    History dropout: during training, with prob `hist_dropout` a sample's whole history is replaced by PAD.
    Reason: past actions are a shortcut ("repeat what I just did"), a known failure of imitation policies
    (copycat / causal confusion).  Dropout forces the model to also work from the image alone.
    """

    def __init__(self, vocab_size, d_model=128, max_text_len=8, n_heads=4, n_layers=4, K=4, H=4,
                 patch=6, grid_hw=(12, 14), n_aux=6, aux_classes=12, dropout=0.0, hist_dropout=0.3):
        super().__init__()
        self.K, self.H, self.patch, self.grid_hw, self.hist_dropout = K, H, patch, grid_hw, hist_dropout
        Hh, Ww = grid_hw
        # vision
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)
        self.patch_mlp = nn.Sequential(nn.GELU(), nn.Linear(d_model, d_model))
        self.patch_norm = nn.LayerNorm(d_model)
        self.row_emb = nn.Parameter(torch.randn(Hh, d_model) * 0.5)
        self.col_emb = nn.Parameter(torch.randn(Ww, d_model) * 0.5)
        # language / time
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.text_pos_emb = nn.Parameter(torch.randn(max_text_len, d_model) * 0.5)
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, d_model))
        # action history (ids 0..2 real, 3 / 3 / 2 = pad)
        if H > 0:
            self.h_dx, self.h_dy = nn.Embedding(4, d_model), nn.Embedding(4, d_model)
            self.h_grip = nn.Embedding(3, d_model)
            self.h_age = nn.Parameter(torch.randn(H, d_model) * 0.5)    # slot H-1 = most recent
            self.register_buffer("pad_id", torch.tensor([PAD_DX, PAD_DY, PAD_GRIP]), persistent=False)
        # chunk queries
        self.query = nn.Parameter(torch.randn(K, d_model) * 0.5)
        # transformer
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
                                                 enable_nested_tensor=False)
        # heads (shared across the K steps)
        self.dx_head, self.dy_head, self.grip_head = nn.Linear(d_model, 3), nn.Linear(d_model, 3), nn.Linear(d_model, 2)
        self.n_aux, self.aux_classes = n_aux, aux_classes
        self.aux_head = nn.Linear(d_model, n_aux * aux_classes)
        # dense grounding: every IMAGE token says 'am I the end-effector / the target block / part of the target region?'
        self.grid_head = nn.Linear(d_model, 3)

    def encode(self, image, text_tokens, time_step, hist=None):
        """Run the transformer; returns ALL token outputs (B, K + 168 + 8 + 1 + H, d_model).
        The first K tokens are the chunk queries, the next 168 the image tokens."""
        B = image.shape[0]
        Hh, Ww = self.grid_hw
        assert image.shape[-2:] == (Hh * self.patch, Ww * self.patch), f"unexpected image size {tuple(image.shape)}"
        x = self.patch_norm(self.patch_mlp(self.patch_embed(image).flatten(2).transpose(1, 2)))
        pos = (self.row_emb[:, None, :] + self.col_emb[None, :, :]).reshape(Hh * Ww, -1)
        parts = [self.query.unsqueeze(0).expand(B, -1, -1),
                 x + pos.unsqueeze(0),
                 self.word_emb(text_tokens) + self.text_pos_emb.unsqueeze(0),
                 self.time_mlp(time_step.float().unsqueeze(-1)).unsqueeze(1)]
        if self.H > 0:
            if hist is None:
                hist = self.pad_id.expand(B, self.H, 3)
            elif self.training and self.hist_dropout > 0:
                drop = torch.rand(B, device=hist.device) < self.hist_dropout
                hist = torch.where(drop[:, None, None], self.pad_id.expand(B, self.H, 3), hist)
            parts.append(self.h_dx(hist[..., 0]) + self.h_dy(hist[..., 1]) + self.h_grip(hist[..., 2])
                         + self.h_age.unsqueeze(0))
        return self.transformer(torch.cat(parts, dim=1))

    def forward(self, image, text_tokens, time_step, hist=None):
        """image (B,3,72,84) float; text_tokens (B,8) long; time_step (B,) float; hist (B,H,3) long or None."""
        B = image.shape[0]
        Hh, Ww = self.grid_hw
        out = self.encode(image, text_tokens, time_step, hist)
        q = out[:, :self.K]                                   # (B, K, d)
        img_out = out[:, self.K:self.K + Hh * Ww]                 # (B, 168, d): the image tokens AFTER the transformer
        return {"dx": self.dx_head(q), "dy": self.dy_head(q), "grip": self.grip_head(q),
                "aux": self.aux_head(q.mean(1)).view(B, self.n_aux, self.aux_classes),
                "grid": self.grid_head(img_out).transpose(1, 2)}    # (B, 3, 168): channels = [ee, target block, target region]