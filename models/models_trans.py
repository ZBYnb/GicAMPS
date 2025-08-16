import torch
from torch import nn
class Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        # 更新展平后的输入维度
        flattened_input_dim = input_dim + condition_dim  # 不再乘以sequence_length
        self.linear = nn.Linear(flattened_input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        # c 已是一个合适的维度，不需要expand
        # 连接 x 和 c
        combined_input = torch.cat((x, c), dim=1)  # 确保x和c的第一维(batch_size)相同
        h = torch.relu(self.linear(combined_input))
        return self.mu(h), self.log_var(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, input_dim):  # 注意这里修改了参数名称为input_dim
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim  # 添加input_dim作为实例变量
        self.linear = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)  # 使用input_dim

    def forward(self, z, c):
        zc_combined = torch.cat((z, c), dim=1)
        h = torch.relu(self.linear(zc_combined))
        reconstruction = torch.sigmoid(self.out(h))
        # 假设reconstruction应该reshape为(batch_size, input_dim)形式
        return reconstruction.view(-1, self.input_dim)  # 使用self.input_dim保证输出形状正确


class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim, input_dim)  # 确保传递input_dim

    def forward(self, x, c):
        mu, log_var = self.encoder(x, c)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return self.decoder(z, c), mu, log_var



