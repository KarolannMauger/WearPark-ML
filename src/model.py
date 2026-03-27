"""CNN 1D model for Parkinson tremor detection.

Architecture:
    Lightweight residual 1D CNN designed for wrist IMU signals sampled at 100 Hz.
    Three residual blocks with decreasing kernel sizes capture tremor patterns
    across multiple frequency scales (3-7 Hz Parkinsonian range).

Input shape:  (batch, 6, 1000) — single wrist, 10 s @ 100 Hz
Output shape: (batch, 1)       — raw logit; apply sigmoid for probability

Specs:
    ~170 K trainable parameters
    Kernel sizes: 11 → 7 → 5 (coarse to fine temporal resolution)
    AdaptiveAvgPool1d makes the model length-agnostic at inference time
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """Residual block for 1D convolutional networks.

    Applies two consecutive Conv1D layers with a skip connection.
    When input and output channels differ, a 1x1 projection aligns dimensions.

    Structure::

        x ─► Conv ─► BN ─► ReLU ─► Dropout ─► Conv ─► BN ─► (+skip) ─► ReLU ─► out

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int): Convolutional kernel size (same padding applied).
        dropout (float): Dropout probability between the two convolutions.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.3):
        super().__init__()
        pad = kernel_size // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )

        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual addition.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_ch, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_ch, length).
        """
        return self.relu(self.conv_block(x) + self.skip(x))


class WearParkCNN1D(nn.Module):
    """Residual 1D CNN for binary Parkinson tremor classification.

    Stacks an input projection followed by three residual blocks and a
    global average pooling layer, then feeds into an MLP classifier.

    Args:
        n_channels (int): Number of input sensor channels. Defaults to 6
            (AccX, AccY, AccZ, GyroX, GyroY, GyroZ).
        seq_len (int): Input sequence length. Defaults to 1000 (10 s @ 100 Hz).
            Informational only — AdaptiveAvgPool1d handles variable lengths.
        dropout (float): Dropout probability inside residual blocks. Defaults to 0.3.
        fc_dropout (float): Dropout probability in the MLP head. Defaults to 0.5.

    Attributes:
        input_proj (nn.Sequential): Initial channel projection (6 → 16).
        res_blocks (nn.Sequential): Three residual blocks with max-pooling.
        gap (nn.AdaptiveAvgPool1d): Global average pooling to a single time step.
        classifier (nn.Sequential): Two-layer MLP producing a single logit.

    Example:
        >>> model = WearParkCNN1D()
        >>> x = torch.randn(16, 6, 1000)  # batch of 16 windows
        >>> logits = model(x)             # shape: (16, 1)
        >>> probs = model.predict_proba(x)
    """

    def __init__(
        self,
        n_channels: int = 6,
        seq_len: int = 1000,
        dropout: float = 0.3,
        fc_dropout: float = 0.5,
    ):
        super().__init__()

        # Initial projection: expand channels before residual blocks
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            # Block 1: 16 → 32, kernel 11 — captures low-freq tremor (~3–7 Hz)
            ResBlock1D(16, 32, kernel_size=11, dropout=dropout),
            nn.MaxPool1d(2),   # length: 1000 → 500

            # Block 2: 32 → 64, kernel 7
            ResBlock1D(32, 64, kernel_size=7, dropout=dropout),
            nn.MaxPool1d(2),   # length: 500 → 250

            # Block 3: 64 → 64, kernel 5 — fine-grained temporal patterns
            ResBlock1D(64, 64, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(2),   # length: 250 → 125
        )

        # Collapse time dimension regardless of input length
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a full forward pass.

        Args:
            x (torch.Tensor): IMU signal of shape (batch, 6, length).

        Returns:
            torch.Tensor: Raw logit of shape (batch, 1).
                Apply ``torch.sigmoid`` to convert to probability.
        """
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.gap(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return Parkinson probability in [0, 1].

        Args:
            x (torch.Tensor): IMU signal of shape (batch, 6, length).

        Returns:
            torch.Tensor: Probability tensor of shape (batch, 1).
        """
        return torch.sigmoid(self.forward(x))

    def count_params(self) -> int:
        """Print and return the number of trainable parameters.

        Returns:
            int: Total trainable parameter count.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total:,}")
        return total


# ---------------------------------MAIN------------------------------------------
if __name__ == "__main__":
    model = WearParkCNN1D()
    model.count_params()
    dummy = torch.randn(16, 6, 1000)
    out = model(dummy)
    print(f"Input: {dummy.shape}  →  Output: {out.shape}")
