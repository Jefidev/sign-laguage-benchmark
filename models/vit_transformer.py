import torch
import torch.nn as nn
from einops import repeat

# https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632


class PositionnalEncoding(nn.Module):
    def __init__(self, embedding_size: int, seq_size: int):
        super(PositionnalEncoding, self).__init__()

        # Classification token put in front of the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))

        # Positional encoding.

        self.positions = nn.Parameter(torch.randn(1, seq_size + 1, embedding_size))
        self.projection = nn.Sequential(nn.Linear(130, embedding_size))

    def forward(self, x):
        """
        Shape of x : Ba
        """
        b, n, _ = x.shape

        x = self.projection(x)

        # Add the classification token
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.positions[:, : (n + 1)]

        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()

        self.clf = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        cls = self.clf(x)
        return cls


class TripletClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int, final_size: int):
        super().__init__()

        self.clf = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, final_size),
        )

    def forward(self, x):
        cls = self.clf(x)
        return cls


class PoseViT(nn.Module):
    """
    Transformer architecture inspired by the ViT paper. Modified to process the pose data of
    the LSFB dataset.
    """

    def __init__(
        self,
        n_classes: int,
        embedding_size: int,
        seq_size: int,
        triplet=False,
        num_layer=2,
        nhead=4,
        triplet_embedding=512,
    ):
        super(PoseViT, self).__init__()

        self.embedding = PositionnalEncoding(
            embedding_size=embedding_size, seq_size=seq_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

        self.to_latent = nn.Identity()

        if triplet:
            self.clf = TripletClassificationHead(
                emb_size=embedding_size, final_size=triplet_embedding
            )
        else:
            self.clf = ClassificationHead(embedding_size, n_classes)

        self.pool = "clf_token"

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.encoder(x, src_key_padding_mask=src_mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        x = self.clf(x)
        return x
