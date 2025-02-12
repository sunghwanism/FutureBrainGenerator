import torch
import torch.nn as nn



def convBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, last_block=False):
    if last_block:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels),
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )



class BrainAgePrediction(nn.Module):

    def __init__(self, config, input_dim=16, latent_shape=(24,28,24)):
        super(BrainAgePrediction, self).__init__()
        num_channels = config.diff_num_channels
        final_shape = (num_channels[2], latent_shape[0]//4, latent_shape[1]//4, latent_shape[2]//4)

        self.Block1 = convBlock(input_dim, num_channels[0])
        self.Block2 = convBlock(num_channels[0], num_channels[1])
        self.Block3 = convBlock(num_channels[1], num_channels[2], last_block=True)
        self.fc = nn.Linear(*final_shape, 1)


    def forward(self, x):
        
        h1 = self.Block1(x)
        h2 = self.Block2(h1)
        h3 = self.Block3(h2)

        x = self.fc(h3.view(h3.size(0), -1))

        return x, [h1, h2, h3]
    