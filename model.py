import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None):
        if state is None:
            batch_size, _, height, width = x.size()
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: list[int], kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = in_channels
        for h_dim in hidden_dims:
            self.layers.append(ConvLSTMCell(prev_dim, h_dim, kernel_size))
            prev_dim = h_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _, height, width = x.size()
        current_input = x

        for layer in self.layers:
            output_inner = []
            state = None
            for t in range(time_steps):
                h, c = layer(current_input[:, t, :, :, :], state)
                state = (h, c)
                output_inner.append(h)
            current_input = torch.stack(output_inner, dim=1)

        return current_input


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatiotemporalUNetConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: list[int] = [64, 128, 256],
        lstm_hidden: list[int] = [512, 512],
    ):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc2 = DoubleConv3D(features[0], features[1])
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc3 = DoubleConv3D(features[1], features[2])
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv_lstm = ConvLSTM(in_channels=features[2], hidden_dims=lstm_hidden)

        self.up3 = nn.ConvTranspose3d(
            lstm_hidden[-1], features[2], kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec3 = DoubleConv3D(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose3d(
            features[2], features[1], kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec2 = DoubleConv3D(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose3d(
            features[1], features[0], kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec1 = DoubleConv3D(features[0] * 2, features[0])

        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b_in = p3.permute(0, 2, 1, 3, 4)
        b_lstm = self.conv_lstm(b_in)
        b_out = b_lstm.permute(0, 2, 1, 3, 4)

        d3 = self.up3(b_out)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)