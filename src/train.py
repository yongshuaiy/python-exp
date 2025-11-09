import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import create_model
from src.utils import set_seed, save_training_curve, ensure_results_dir,create_transformer_masks,create_lm_mask
from src.data_loader import load_data
from src.config import load_config, get_device


class Trainer:
    def __init__(self, model, train_loader, val_loader, vocab_size, config, model_name='transformer'):
        self.model = model.to(get_device(config))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.config = config
        self.model_name = model_name

        self.patience = 3  # 减少耐心值
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stop = False

        # 跟踪最佳模型状态
        self.best_model_state = None
        self.vocab = None  # 稍后设置词汇表

        training_config = config['training']

        # 确保数值类型正确
        lr = float(training_config['learning_rate'])
        weight_decay = float(training_config['weight_decay'])

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98),eps=1e-9)
        self.scheduler = None  # 不使用任何调度器

        # 或者使用简单的StepLR（可选）
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        # 添加学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(training_config.get('epochs', 10)) * len(train_loader),
            eta_min=lr * 0.1  # 最小学习率为初始学习率的10%
        )
        # 改进的学习率调度器
        if training_config.get('use_warmup', False):
            # 使用带预热的调度器
            warmup_steps = int(training_config.get('warmup_steps', 1000))
            total_steps = int(training_config.get('epochs', 10)) * len(train_loader)

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            # 使用余弦退火
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(training_config.get('epochs', 10)) * len(train_loader),
                eta_min=lr * 0.01  # 最小学习率为初始学习率的1%
            )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        ensure_results_dir()

    def set_vocab(self, vocab):
        """设置词汇表用于文本解码"""
        self.vocab = vocab

    def generate_and_compare(self, num_examples=5, max_length=20):
        """生成文本并与真实文本对比"""
        if self.vocab is None:
            print("Warning: Vocabulary not set. Cannot generate text.")
            return []  # 返回空列表而不是None

        self.model.eval()
        examples = []

        try:
            with torch.no_grad():
                for i, (data, target) in enumerate(self.val_loader):
                    if i >= num_examples:
                        break

                    # 使用源序列作为输入，目标序列作为参考
                    src_sequence = data[:1]  # 取第一个样本作为源序列

                    # 检查模型是否有generate方法
                    if not hasattr(self.model, 'generate'):
                        print("Warning: Model doesn't have generate method")
                        return []

                    # 生成文本
                    generated = self.model.generate(
                        src_sequence,
                        max_length=max_length,
                        temperature=0.8,
                        top_k=50
                    )

                    # 解码文本
                    src_text = self._decode_tokens(src_sequence[0])
                    generated_text = self._decode_tokens(generated[0])
                    target_text = self._decode_tokens(target[0])

                    # 计算相似度
                    similarity = self._calculate_similarity(generated_text, target_text)

                    examples.append({
                        'source': src_text,
                        'generated': generated_text,
                        'target': target_text,
                        'similarity': similarity,
                        'source_tokens': src_sequence[0].cpu().tolist(),
                        'generated_tokens': generated[0].cpu().tolist(),
                        'target_tokens': target[0].cpu().tolist()
                    })

            return examples

        except Exception as e:
            print(f"Error during text generation: {e}")
            return []  # 发生错误时返回空列表

    def _decode_tokens(self, tokens):
        """将token IDs解码为文本"""
        if hasattr(self.vocab, 'get_itos'):
            # torchtext词汇表
            return ' '.join([self.vocab.get_itos()[token] for token in tokens if token != 0])
        elif hasattr(self.vocab, 'itos'):
            # 自定义词汇表
            return ' '.join([self.vocab.itos[token] for token in tokens if token != 0])
        else:
            # 简单词汇表（使用字母表示）
            return ' '.join([f'token_{token}' for token in tokens if token != 0])

    def _calculate_similarity(self, text1, text2):
        """计算两个文本的简单相似度"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def print_generation_examples(self, examples):
        """打印生成示例"""
        if not examples:
            print("没有可用的生成示例")
            return 0.0

        print("\n" + "=" * 80)
        print("文本生成对比结果")
        print("=" * 80)

        for i, example in enumerate(examples):
            print(f"\n示例 {i + 1}:")
            print(f"源序列:    {example['source']}")
            print(f"生成序列:  {example['generated']}")
            print(f"目标序列:  {example['target']}")
            print(f"相似度:    {example['similarity']:.4f}")
            print(f"源Tokens:  {example['source_tokens']}")
            print(f"生成Tokens:{example['generated_tokens']}")
            print(f"目标Tokens:{example['target_tokens']}")

        # 计算平均相似度
        avg_similarity = sum(ex['similarity'] for ex in examples) / len(examples)
        print(f"\n平均相似度: {avg_similarity:.4f}")

        return avg_similarity

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc='Training')

        clip_grad_norm = float(self.config['training']['clip_grad_norm'])
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(get_device(self.config)), target.to(get_device(self.config))

            self.optimizer.zero_grad()

            if hasattr(self.model, 'encoder'):  # 完整Transformer
                src_mask, tgt_mask, memory_mask = create_transformer_masks(data, data[:, :-1])
                # 解码器输入是去掉最后一个token
                decoder_input = data[:, :-1]
                # 目标应该是从第二个token开始，与解码器输出对齐
                output_target = target[:, 1:]
                output = self.model(data, decoder_input, src_mask, tgt_mask, memory_mask)
                loss = self.criterion(output.view(-1, self.vocab_size), output_target.contiguous().view(-1))
            else:  # 语言模型
                mask = create_lm_mask(data)
                output = self.model(data, mask)
                loss = self.criterion(output.view(-1, self.vocab_size), target.view(-1))
            # output = self.model(data)
            # loss = self.criterion(output.view(-1, self.vocab_size), target.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad_norm)
            self.optimizer.step()

            # 更新学习率
            self.scheduler.step()

            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(get_device(self.config)), target.to(get_device(self.config))
                if hasattr(self.model, 'encoder'):  # 完整Transformer
                    src_mask, tgt_mask, memory_mask = create_transformer_masks(data, data[:, :-1])

                    # 对于自回归任务，使用输入的前n-1个token作为decoder输入
                    decoder_input = data[:, :-1]
                    output_target = target[:, 1:]
                    output = self.model(data, decoder_input, src_mask, tgt_mask, memory_mask)
                    # output = self.model(data, decoder_input)
                    loss = self.criterion(output.view(-1, self.vocab_size), output_target.contiguous().view(-1))
                else:  # 语言模型
                    mask = create_lm_mask(data)
                    output = self.model(data, mask)
                    # output = self.model(data)
                    loss = self.criterion(output.view(-1, self.vocab_size), target.view(-1))
                # output = self.model(data)
                # loss = self.criterion(output.view(-1, self.vocab_size), target.view(-1))

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs=None):
        ensure_results_dir()

        if epochs is None:
            epochs = int(self.config['training']['epochs'])

        best_val_loss = float('inf')
        save_path = os.path.join(self.config['experiment']['save_dir'], f'best_{self.model_name}.pth')

        # 记录生成质量
        generation_qualities = []

        for epoch in range(epochs):
            # 检查是否应该早停
            if self.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - start_time

            print(f'Epoch {epoch + 1}/{epochs} | Time: {epoch_time:.2f}s | '
                  f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                  f'LR: {self.learning_rates[-1]:.2e}')
            # 改进的模型保存逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # 保存最佳模型状态
                self.best_model_state = {
                    'model_state_dict': self.model.state_dict().copy(),
                    'optimizer_state_dict': self.optimizer.state_dict().copy(),
                    'scheduler_state_dict': self.scheduler.state_dict().copy(),
                    'config': self.config,
                    'val_loss': val_loss,
                    'epoch': epoch
                }
                torch.save(self.best_model_state, save_path)
                print(f"Best model saved to {save_path} (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epochs (best: {best_val_loss:.4f})")

                # 动态调整学习率
                if self.epochs_without_improvement >= 2:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = current_lr * 0.5  # 学习率减半
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Reduced learning rate to {new_lr:.2e}")

                # 检查是否应该早停
                if self.epochs_without_improvement >= self.patience:
                    self.early_stop = True
                    print("Early stopping triggered")

                    # 恢复最佳模型
                    if self.best_model_state:
                        self.model.load_state_dict(self.best_model_state['model_state_dict'])
                        print("Restored best model weights")

            # 每隔几个epoch进行一次生成评估
            if epoch % 5 == 0 or epoch == epochs - 1:
                print("\n正在进行文本生成评估...")
                examples = self.generate_and_compare(num_examples=3, max_length=20)

                # 检查examples是否有效
                if examples:
                    avg_similarity = self.print_generation_examples(examples)
                    generation_qualities.append((epoch, avg_similarity))
                    # 保存生成示例
                    self.save_generation_examples(examples, epoch)
                else:
                    print("文本生成评估失败或没有生成示例")
                    generation_qualities.append((epoch, 0.0))  # 记录0相似度

            # # 检查验证损失是否改善
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save({
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': self.optimizer.state_dict(),
            #         'scheduler_state_dict': self.scheduler.state_dict(),
            #         'config': self.config,
            #         'val_loss': val_loss,
            #         'epoch': epoch
            #     }, save_path)
            #     print(f"Best model saved to {save_path}")
            # else:
            #     self.epochs_without_improvement += 1
            #     print(f"No improvement for {self.epochs_without_improvement} epochs")
            #
            #     # 检查是否应该早停
            #     if self.epochs_without_improvement >= self.patience:
            #         self.early_stop = True
            #         print("Early stopping triggered")

        curve_filename = f'{self.model_name}_training_curve.png'
        save_training_curve(self.train_losses, self.val_losses, curve_filename)

        # 保存学习率曲线
        self.save_learning_rate_curve()

        return best_val_loss

    def save_learning_rate_curve(self):
        """保存学习率变化曲线"""
        ensure_results_dir()
        save_path = os.path.join(self.config['experiment']['save_dir'], f'{self.model_name}_learning_rate.png')

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.learning_rates)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.savefig(save_path)
        plt.close()
        print(f"Learning rate curve saved to {save_path}")


def save_generation_examples(self, examples, epoch):
    """保存生成示例到文件"""
    ensure_results_dir()
    filename = f'{self.model_name}_generation_epoch_{epoch}.txt'
    filepath = os.path.join(self.config['experiment']['save_dir'], filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"生成示例 - Epoch {epoch}\n")
        f.write("=" * 50 + "\n\n")

        for i, example in enumerate(examples):
            f.write(f"示例 {i + 1}:\n")
            f.write(f"源序列:    {example['source']}\n")
            f.write(f"生成序列:  {example['generated']}\n")
            f.write(f"目标序列:  {example['target']}\n")
            f.write(f"相似度:    {example['similarity']:.4f}\n")
            f.write(f"源Tokens:  {example['source_tokens']}\n")
            f.write(f"生成Tokens:{example['generated_tokens']}\n")
            f.write(f"目标Tokens:{example['target_tokens']}\n\n")

    print(f"生成示例保存到: {filepath}")


def save_generation_quality_curve(self, generation_qualities):
    """保存生成质量变化曲线"""
    if not generation_qualities:
        return

    ensure_results_dir()
    save_path = os.path.join(self.config['experiment']['save_dir'],
                             f'{self.model_name}_generation_quality.png')

    epochs, qualities = zip(*generation_qualities)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, qualities, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Generation Similarity')
    plt.title('Text Generation Quality Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

    print(f"生成质量曲线保存到: {save_path}")

def main():
    # 加载配置
    config = load_config()

    # 设置随机种子
    set_seed(int(config['experiment']['seed']))

    # 检查设备
    device = get_device(config)
    print(f'Using device: {device}')
    print(f'Configuration: {config}')

    # 加载数据
    train_loader, val_loader, vocab = load_data(config)

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # 创建模型
    print("Creating model...")
    model = create_model(vocab_size, config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器并开始训练
    trainer = Trainer(model, train_loader, val_loader, vocab_size, config, 'transformer')
    print("Starting training...")
    best_loss = trainer.train()

    print(f"Training completed! Best validation loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()