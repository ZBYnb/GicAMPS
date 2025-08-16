import torch
from torch import nn

# Encoder Definition
class Encoder(nn.Module):
    def __init__(self, sequence_length, input_dim, condition_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        # 计算展平后的输入维度flattened_input_dim = 930
        flattened_input_dim = (input_dim + condition_dim) * sequence_length
        self.linear = nn.Linear(flattened_input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        c = c.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, sequence_length, condition_dim)
        # 连接 x 和 c
        x = torch.cat((x, c), dim=2)  # (batch_size, sequence_length, input_dim + condition_dim)
        # 展平 x，使其能够输入到全连接层
        x = x.view(x.size(0), -1)  # (batch_size, flattened_input_dim)

        # 通过全连接层
        h = torch.relu(self.linear(x))
        return self.mu(h), self.log_var(h)

# Decoder Definition
class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, sequence_length, input_dim):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        # 线性层输入维度应为 latent_dim + condition_dim
        self.linear = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, sequence_length * input_dim)

    def forward(self, z, c):
        # 假设 z 的形状为 (batch_size, latent_dim)
        # c 的形状为 (batch_size, condition_dim)

        # 连接 z 和 c
        z = torch.cat((z, c), dim=1)  # (batch_size, latent_dim + condition_dim)

        # 通过全连接层
        h = torch.relu(self.linear(z))
        x = torch.sigmoid(self.out(h))

        # 还原形状为原始输入的形状
        x = x.view(-1, self.sequence_length, self.input_dim)  # (batch_size, sequence_length, input_dim)
        return x

# CVAE Definition
class CVAE(nn.Module):
    def __init__(self, sequence_length, input_dim, condition_dim, hidden_dim, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(sequence_length, input_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim, sequence_length, input_dim)

    def forward(self, x, c):
        #获取均值和方差
        mu, log_var = self.encoder(x, c)

        #方差还原为标准差
        std = torch.exp(0.5 * log_var)

        #生成一组随机的变量，从标准正态分布中采样
        eps = torch.randn_like(std)

        #重参数化
        z = eps * std + mu
        return self.decoder(z, c), mu, log_var


