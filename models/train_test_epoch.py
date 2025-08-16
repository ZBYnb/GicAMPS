import os
import torch
from CVAE.base.loss import loss_function
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def decode_one_hot_amp(one_hot_sequence):
    idx_to_aa = index_to_amino_acid()  # 获取索引到氨基酸的映射
    indices = one_hot_sequence.argmax(dim=-1)  # 从独热编码中获得最大值的索引
    # 确保处理的是一维数组
    if indices.dim() > 1:
        indices = indices.squeeze()  # 去除单维条目
    sequence = ''.join(idx_to_aa.get(idx.item(), 'X') for idx in indices)  # 将索引转换为氨基酸序列
    sequence = sequence.rstrip('X')  # 去掉序列末尾的填充字符
    return sequence


def index_to_amino_acid():
    idx_to_aa = 'ACDEFGHIKLMNPQRSTVWY' + 'X'  # 添加'X'作为未知的填充字符
    return {idx: aa for idx, aa in enumerate(idx_to_aa)}

def decode_one_hot(one_hot_sequence):
    idx_to_aa = index_to_amino_acid()  # 获取索引到氨基酸的映射
    indices = one_hot_sequence.argmax(dim=-1)  # 从独热编码中获得最大值的索引
    sequence = ''.join(idx_to_aa.get(idx.item(), 'X') for idx in indices)  # 将索引转换为氨基酸序列
    sequence = sequence.rstrip('X')  # 去掉序列末尾的填充字符
    return sequence


def train(model, train_loader, val_loader, optimizer, epochs=1000, visualize_every=50, save_every=100,
          save_path='models_saved'):
    model.to(device)
    # 确保保存模型的目录存在
    os.makedirs(save_path, exist_ok=True)
    best_val_loss = float('inf')  # 初始化最佳验证损失

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_samples = 0
        for data, conditions in train_loader:
            data, conditions = data.to(device), conditions.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data, conditions)
            loss = loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

        average_train_loss = total_train_loss / total_samples

        # Evaluate on validation set
        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            for data, conditions in val_loader:
                data, conditions = data.to(device), conditions.to(device)
                recon_batch, mu, log_var = model(data, conditions)
                val_loss = loss_function(recon_batch, data, mu, log_var)
                total_val_loss += val_loss.item() * data.size(0)
                total_val_samples += data.size(0)

        average_val_loss = total_val_loss / total_val_samples



        # Save model every 'save_every' epochs
        if epoch % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
            print(f'Model saved at epoch {epoch+1}')
            # Print training and validation loss
            print(f"Epoch {epoch + 1}: Train Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}")

        # Save the best model if the current validation loss is the lowest
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"Best model updated at epoch {epoch+1} with Validation Loss: {average_val_loss:.4f}")

        # Visualization
        if epoch % visualize_every == 0:
            original_seq = decode_one_hot(data[0])  # 解码第一个样本的原始序列
            generated_seq = decode_one_hot(recon_batch[0])  # 解码第一个样本的生成序列
            print(f"Visualization at Epoch {epoch+1}:")
            print(f"Original Sequence: {original_seq}")
            print(f"Generated Sequence: {generated_seq}")

