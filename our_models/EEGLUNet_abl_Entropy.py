import torch
import torch.nn as nn

class SpatialFeatureExtractor(nn.Module):
    """
    SpatialFeatureExtractor: Applies spatial convolutions to capture inter-channel dependencies.

    Args:
        n_chans (int): Number of input channels.
        spatial_expansion (int): Expansion factor for spatial feature maps.

    Returns:
        Processed spatial feature tensor.
    """

    def __init__(self, n_chans, spatial_expansion):
        super().__init__()

        spatial_convs = [i for i in [4, 8] if i < n_chans] + [n_chans]
        feature_dims = [(spatial_expansion // (n_chans - s_c + 1), s_c) for s_c in spatial_convs]
        self.spatial_filters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, dim[0], kernel_size=(dim[1], 1)))
            for dim in feature_dims
        ])

    def forward(self, input):
        batch_size, _, input_window_samples = input.shape
        input = input.unsqueeze(1)  # Add a channel dimension for convolution
        spatial_features = [conv(input).reshape(batch_size, -1, input_window_samples) for conv in self.spatial_filters]
        Xs = torch.cat(spatial_features, 1).unsqueeze(2)
        return Xs


class TemporalFeatureExtractor(nn.Module):
    """
    TemporalFeatureExtractor: Captures temporal dependencies by applying grouped convolutions.

    Args:
        spatial_merge (int): Dimension of merged spatial feature.
        filters (list): List of filter sizes to apply.

    Returns:
        Processed temporal feature tensor.
    """

    def __init__(self, spatial_merge, filters):
        super().__init__()
        self.temporal_filters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(spatial_merge, spatial_merge, kernel_size=(1, size), padding='same', groups=spatial_merge),
                nn.BatchNorm2d(spatial_merge),
            ) for size in filters
        ])

    def forward(self, Xs):
        Xt = torch.stack([conv(Xs) for conv in self.temporal_filters], dim=1)
        Xt = Xt.reshape(Xt.shape[0], -1, Xt.shape[-1])
        return Xt


class EEGLUNet_abl_Entropy(nn.Module):

    def __init__(self, n_chans: int, n_class: int = 4, spatial_expansion: int = 32, spatial_merge: int = 32, filters: list = None):
        super().__init__()
        # Set default temporal convolution kernel sizes if not specified
        filters = filters or [65, 70, 75]

        self.global_local_spatial = SpatialFeatureExtractor(n_chans, spatial_expansion)

        spatial_convs = [i for i in [4, 8] if i < n_chans] + [n_chans]
        feature_dims = [(spatial_expansion // (n_chans - s_c + 1), s_c) for s_c in spatial_convs]
        s_len = sum(dim[0] * (n_chans - dim[1] + 1) for dim in feature_dims)

        self.filter_bank = TemporalFeatureExtractor(s_len, filters)

        # Fully connected layer for classification
        self.fc = nn.Linear((len(filters) * s_len) ** 2, n_class)

    def lu_space(self, Xb: torch.Tensor) -> torch.Tensor:
        _, L, U = torch.linalg.lu(Xb)

        tril_idx = torch.tril_indices(L.size(-2), L.size(-1), offset=-1)
        L_tril = L[:, tril_idx[0], tril_idx[1]]

        triu_idx = torch.triu_indices(U.size(-2), U.size(-1), offset=0)
        U_triu = U[:, triu_idx[0], triu_idx[1]]

        return torch.cat((L_tril, U_triu), dim=-1)

    def forward(self, input):
        batch_size, _, input_window_samples = input.shape

        # Spatial and Temporal Feature Extraction
        Xs = self.global_local_spatial(input)
        # Xs = self.entropy_attention(Xs)
        Xt = self.filter_bank(Xs)

        Xb = torch.bmm(Xt, Xt.transpose(1, 2)) / (input_window_samples - 1)

        # LU space
        Xb_features = self.lu_space(Xb)

        # Classification
        Xm = self.fc(Xb_features)

        return Xm


if __name__ == '__main__':
    x = torch.randn(70, 22, 1000).to('cpu').float()
    model = EEGLUNet_abl_Entropy(22, 4).to('cpu').float()
    y = model(x)
    print(y.shape)
