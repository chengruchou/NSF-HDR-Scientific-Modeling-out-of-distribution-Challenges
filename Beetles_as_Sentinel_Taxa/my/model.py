import torch
import torch.nn as nn
import torch.nn.functional as F

class GrandBeetleModel(nn.Module):
    def __init__(
        self,
        backbone,
        num_species,
        num_domains,
        backbone_dim=768,
        embedding_dim=64,
        dropout_rate=0.3
    ):
        """
        Args:
            backbone: The pretrained vision model (BioCLIP or DINO).
            num_species: Total unique species count (for embedding).
            num_domains: Total unique domain count (for embedding).
            backbone_dim: Output dimension of the vision backbone (768 for ViT-B).
            embedding_dim: Size of the learnable vectors for species/domain.
        """
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.backbone_dim = backbone_dim

        self.spatial_head = nn.Sequential(
            nn.Conv2d(backbone_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Reduce to fixed 4x4 grid regardless of input size
        )

        # Flattened size: 256 channels * 4 * 4 = 4096 features
        self.vision_out_dim = 256 * 4 * 4

        # --- 2. Context Tower (Metadata) ---
        self.species_embed = nn.Embedding(num_species, embedding_dim)
        self.domain_embed = nn.Embedding(num_domains, embedding_dim)

        # Special token for "Unknown/Padding" domain (for dropout)
        self.padding_idx = 0

        # --- 3. Fusion & Regressor ---
        fusion_dim = self.vision_out_dim + (embedding_dim * 2)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 3) # Output: SPEI_30d, SPEI_1y, SPEI_2y
        )

    def forward(self, images, species_idx, domain_idx, domain_dropout_prob=0.0):
        """
        Args:
            images: Batch of images (or pre-extracted feature maps if freezing).
            species_idx: Tensor of species integers.
            domain_idx: Tensor of domain integers.
            domain_dropout_prob: Probability to mask domain info during training.
        """
        x = self.spatial_head(images)
        x = x.reshape(x.size(0), -1) # Flatten to (Batch, 4096)

        if self.training and domain_dropout_prob > 0:
            mask = torch.rand_like(domain_idx.float()) < domain_dropout_prob
            domain_idx = torch.where(mask, torch.tensor(0, device=domain_idx.device), domain_idx)

        sp_emb = self.species_embed(species_idx)
        dom_emb = self.domain_embed(domain_idx)

        combined = torch.cat([x, sp_emb, dom_emb], dim=1)

        return self.regressor(combined)