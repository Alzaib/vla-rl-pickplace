import torch
import torch.nn as nn


class SimpleVLA(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        max_text_len=8,
        n_heads=4,
        n_layers=4,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_text_len = max_text_len

        # ==========================================================
        # Vision Encoder
        # 72x84 -> 36x42 -> 18x21 -> 9x11 -> 5x6
        # ==========================================================

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 6))

        # 30 image positions
        self.image_pos_emb = nn.Parameter(
            torch.randn(30, d_model)
        )

        # ==========================================================
        # Language Encoder
        # ==========================================================

        self.word_emb = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.text_pos_emb = nn.Parameter(
            torch.randn(max_text_len, d_model)
        )

        # ==========================================================
        # Time Token
        # ==========================================================

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        # ==========================================================
        # ACT Token
        # ==========================================================

        self.act_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        # ==========================================================
        # Transformer
        # ==========================================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # ==========================================================
        # Action Heads
        # ==========================================================

        self.dx_head = nn.Linear(d_model, 3)
        self.dy_head = nn.Linear(d_model, 3)
        self.grip_head = nn.Linear(d_model, 2)

    def forward(self, image, text_tokens, time_step):

        B = image.shape[0]

        # ==========================================================
        # IMAGE TOKENS
        # ==========================================================

        x = self.cnn(image)

        x = self.adaptive_pool(x)

        # B x 128 x 5 x 6
        x = x.flatten(2)

        # B x 128 x 30
        x = x.transpose(1, 2)

        # B x 30 x 128
        image_tokens = x + self.image_pos_emb.unsqueeze(0)

        # ==========================================================
        # TEXT TOKENS
        # ==========================================================

        text_emb = self.word_emb(text_tokens)

        text_emb = (
            text_emb
            + self.text_pos_emb.unsqueeze(0)
        )

        # ==========================================================
        # TIME TOKEN
        # ==========================================================

        time_step = time_step.float().unsqueeze(-1)

        time_token = self.time_mlp(time_step)

        time_token = time_token.unsqueeze(1)

        # B x 1 x 128

        # ==========================================================
        # ACT TOKEN
        # ==========================================================

        act_token = self.act_token.repeat(B, 1, 1)

        # ==========================================================
        # CONCAT TOKENS
        # ==========================================================

        tokens = torch.cat(
            [
                act_token,
                image_tokens,
                text_emb,
                time_token,
            ],
            dim=1,
        )

        # Shape:
        #
        # 1 ACT
        # 30 image
        # 8 text
        # 1 time
        #
        # total = 40 tokens

        # ==========================================================
        # TRANSFORMER
        # ==========================================================

        tokens = self.transformer(tokens)

        act_out = tokens[:, 0]

        # ==========================================================
        # ACTION HEADS
        # ==========================================================

        dx_logits = self.dx_head(act_out)
        dy_logits = self.dy_head(act_out)
        grip_logits = self.grip_head(act_out)

        return {
            "dx": dx_logits,
            "dy": dy_logits,
            "grip": grip_logits,
        }
