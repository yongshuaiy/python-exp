import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json

from src.model import TransformerLM
from src.train import Trainer
from src.data_loader import create_simple_dataset, load_wikitext2_offline
from src.utils import set_seed, ensure_results_dir
from src.config import load_config, get_device


class AblationStudy:
    def __init__(self, config):
        self.config = config
        self.device = get_device(config)
        self.results = {}
        self.full_model_result = None  # 存储完整模型的结果
        ensure_results_dir()

    def run_full_model(self):
        """运行完整模型作为基准"""
        print("Running full model as baseline...")

        data_config = self.config['data']
        vocab_size = int(data_config['vocab_size'])
        num_samples = 2000
        seq_len = int(data_config['seq_len'])
        batch_size = int(data_config['batch_size'])
        epochs = self.config['training']['ablation_epochs'] #20


        # train_loader, val_loader, vocab = create_simple_dataset(
        #     vocab_size=vocab_size, num_samples=num_samples, seq_len=seq_len, batch_size=batch_size
        # )
        # 使用真实数据
        dataset_name = data_config['dataset']
        if dataset_name == 'wikitext2':
            try:
                train_loader, val_loader, vocab = load_wikitext2_offline(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    use_sample_data=False
                )
                vocab_size = len(vocab)
            except Exception as e:
                print(f"真实数据加载失败: {e}, 使用合成数据")
                vocab_size = int(data_config['vocab_size'])
                train_loader, val_loader, vocab = create_simple_dataset(
                    vocab_size=vocab_size,
                    num_samples=2000,
                    seq_len=seq_len,
                    batch_size=batch_size
                )
        else:
            vocab_size = int(data_config['vocab_size'])
            train_loader, val_loader, vocab = create_simple_dataset(
                vocab_size=vocab_size,
                num_samples=2000,
                seq_len=seq_len,
                batch_size=batch_size
            )

        set_seed(int(self.config['experiment']['seed']))
        model = TransformerLM(vocab_size, self.config)

        trainer = Trainer(model, train_loader, val_loader, vocab_size,
                          self.config, 'full_model')
        trainer.train(epochs=epochs)

        self.full_model_result = {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else float('inf')
        }

        return self.full_model_result

    def run_positional_encoding_ablation(self):
        """位置编码消融实验"""
        print("Running positional encoding ablation study...")

        data_config = self.config['data']
        vocab_size = int(data_config['vocab_size'])
        num_samples = 2000
        seq_len = int(data_config['seq_len'])
        batch_size = int(data_config['batch_size'])
        epochs = self.config['training']['ablation_epochs'] #20

        # 使用真实数据
        dataset_name = data_config['dataset']
        if dataset_name == 'wikitext2':
            try:
                train_loader, val_loader, vocab = load_wikitext2_offline(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    use_sample_data=False
                )
                vocab_size = len(vocab)
            except Exception as e:
                print(f"真实数据加载失败: {e}, 使用合成数据")
                vocab_size = int(data_config['vocab_size'])
                train_loader, val_loader, vocab = create_simple_dataset(
                    vocab_size=vocab_size,
                    num_samples=2000,
                    seq_len=seq_len,
                    batch_size=batch_size
                )
        else:
            vocab_size = int(data_config['vocab_size'])
            train_loader, val_loader, vocab = create_simple_dataset(
                vocab_size=vocab_size,
                num_samples=2000,
                seq_len=seq_len,
                batch_size=batch_size
            )

        # 有位置编码的模型
        set_seed(int(self.config['experiment']['seed']))
        model_with_pe = TransformerLM(vocab_size, self.config)

        # 无位置编码的模型 - 创建一个新的配置，将max_seq_len设置为0
        set_seed(int(self.config['experiment']['seed']))
        config_without_pe = self.config.copy()
        config_without_pe['model']['max_seq_len'] = 0  # 设置为0表示不使用位置编码

        model_without_pe = TransformerLM(vocab_size, config_without_pe)

        trainer_with_pe = Trainer(model_with_pe, train_loader, val_loader, vocab_size,
                                  self.config, 'with_positional_encoding')
        trainer_without_pe = Trainer(model_without_pe, train_loader, val_loader, vocab_size,
                                     config_without_pe, 'without_positional_encoding')

        trainer_with_pe.train(epochs=epochs)
        trainer_without_pe.train(epochs=epochs)

        self.results['positional_encoding'] = {
            'with_pe': {
                'val_losses': trainer_with_pe.val_losses,
                'final_val_loss': trainer_with_pe.val_losses[-1] if trainer_with_pe.val_losses else float('inf')
            },
            'without_pe': {
                'val_losses': trainer_without_pe.val_losses,
                'final_val_loss': trainer_without_pe.val_losses[-1] if trainer_without_pe.val_losses else float('inf')
            }
        }

    def run_attention_heads_ablation(self):
        """注意力头数消融实验"""
        print("Running attention heads ablation study...")

        data_config = self.config['data']
        vocab_size = int(data_config['vocab_size'])
        num_samples = 2000
        seq_len = int(data_config['seq_len'])
        batch_size = int(data_config['batch_size'])
        epochs = self.config['training']['ablation_epochs'] #20

        # train_loader, val_loader, vocab = create_simple_dataset(
        #     vocab_size=vocab_size, num_samples=num_samples, seq_len=seq_len, batch_size=batch_size
        # )
        # 使用真实数据
        dataset_name = data_config['dataset']
        if dataset_name == 'wikitext2':
            try:
                train_loader, val_loader, vocab = load_wikitext2_offline(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    use_sample_data=False
                )
                vocab_size = len(vocab)
            except Exception as e:
                print(f"真实数据加载失败: {e}, 使用合成数据")
                vocab_size = int(data_config['vocab_size'])
                train_loader, val_loader, vocab = create_simple_dataset(
                    vocab_size=vocab_size,
                    num_samples=2000,
                    seq_len=seq_len,
                    batch_size=batch_size
                )
        else:
            vocab_size = int(data_config['vocab_size'])
            train_loader, val_loader, vocab = create_simple_dataset(
                vocab_size=vocab_size,
                num_samples=2000,
                seq_len=seq_len,
                batch_size=batch_size
            )

        head_configs = [1, 2, 4, 8]
        results = {}

        for num_heads in head_configs:
            print(f"Training with {num_heads} attention heads...")
            set_seed(int(self.config['experiment']['seed']))

            head_config = self.config.copy()
            head_config['model']['num_heads'] = num_heads

            model = TransformerLM(vocab_size, head_config)

            trainer = Trainer(model, train_loader, val_loader, vocab_size,
                              head_config, f'heads_{num_heads}')
            trainer.train(epochs=epochs)

            results[num_heads] = {
                'val_losses': trainer.val_losses,
                'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else float('inf')
            }

        self.results['attention_heads'] = results

    def run_layer_depth_ablation(self):
        """层数消融实验"""
        print("Running layer depth ablation study...")

        data_config = self.config['data']
        vocab_size = int(data_config['vocab_size'])
        num_samples = 2000
        seq_len = int(data_config['seq_len'])
        batch_size = int(data_config['batch_size'])
        epochs = self.config['training']['ablation_epochs'] #20

        # train_loader, val_loader, vocab = create_simple_dataset(
        #     vocab_size=vocab_size, num_samples=num_samples, seq_len=seq_len, batch_size=batch_size
        # )
        # 使用真实数据
        dataset_name = data_config['dataset']
        if dataset_name == 'wikitext2':
            try:
                train_loader, val_loader, vocab = load_wikitext2_offline(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    use_sample_data=False
                )
                vocab_size = len(vocab)
            except Exception as e:
                print(f"真实数据加载失败: {e}, 使用合成数据")
                vocab_size = int(data_config['vocab_size'])
                train_loader, val_loader, vocab = create_simple_dataset(
                    vocab_size=vocab_size,
                    num_samples=2000,
                    seq_len=seq_len,
                    batch_size=batch_size
                )
        else:
            vocab_size = int(data_config['vocab_size'])
            train_loader, val_loader, vocab = create_simple_dataset(
                vocab_size=vocab_size,
                num_samples=2000,
                seq_len=seq_len,
                batch_size=batch_size
            )

        layer_configs = [1, 2, 4, 6]
        results = {}

        for num_layers in layer_configs:
            print(f"Training with {num_layers} layers...")
            set_seed(int(self.config['experiment']['seed']))

            layer_config = self.config.copy()
            layer_config['model']['num_layers'] = num_layers

            model = TransformerLM(vocab_size, layer_config)

            trainer = Trainer(model, train_loader, val_loader, vocab_size,
                              layer_config, f'layers_{num_layers}')
            trainer.train(epochs=epochs)

            results[num_layers] = {
                'val_losses': trainer.val_losses,
                'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else float('inf')
            }

        self.results['layer_depth'] = results

    def run_embedding_dim_ablation(self):
        """词嵌入维度消融实验"""
        print("Running embedding dimension ablation study...")

        data_config = self.config['data']
        vocab_size = int(data_config['vocab_size'])
        num_samples = 2000
        seq_len = int(data_config['seq_len'])
        batch_size = int(data_config['batch_size'])
        epochs = self.config['training']['ablation_epochs'] #20

        # train_loader, val_loader, vocab = create_simple_dataset(
        #     vocab_size=vocab_size, num_samples=num_samples, seq_len=seq_len, batch_size=batch_size
        # )
        # 使用真实数据
        dataset_name = data_config['dataset']
        if dataset_name == 'wikitext2':
            try:
                train_loader, val_loader, vocab = load_wikitext2_offline(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    use_sample_data=False
                )
                vocab_size = len(vocab)
            except Exception as e:
                print(f"真实数据加载失败: {e}, 使用合成数据")
                vocab_size = int(data_config['vocab_size'])
                train_loader, val_loader, vocab = create_simple_dataset(
                    vocab_size=vocab_size,
                    num_samples=2000,
                    seq_len=seq_len,
                    batch_size=batch_size
                )
        else:
            vocab_size = int(data_config['vocab_size'])
            train_loader, val_loader, vocab = create_simple_dataset(
                vocab_size=vocab_size,
                num_samples=2000,
                seq_len=seq_len,
                batch_size=batch_size
            )

        dim_configs = [64, 128, 256, 512]
        results = {}

        for d_model in dim_configs:
            print(f"Training with embedding dimension {d_model}...")
            set_seed(int(self.config['experiment']['seed']))

            dim_config = self.config.copy()
            dim_config['model']['d_model'] = d_model
            # 调整前馈网络维度以匹配嵌入维度
            dim_config['model']['d_ff'] = d_model * 4

            model = TransformerLM(vocab_size, dim_config)

            trainer = Trainer(model, train_loader, val_loader, vocab_size,
                              dim_config, f'embedding_{d_model}')
            trainer.train(epochs=epochs)

            results[d_model] = {
                'val_losses': trainer.val_losses,
                'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else float('inf')
            }

        self.results['embedding_dim'] = results

    def plot_results(self):
        """绘制消融实验结果"""
        ensure_results_dir()
        save_path = os.path.join(self.config['experiment']['save_dir'], 'ablation_results.png')

        num_studies = len(self.results)
        if num_studies == 0:
            print("No ablation results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        studies = list(self.results.keys())

        for i, study_name in enumerate(studies):
            if i >= len(axes):
                break

            ax = axes[i]
            study_data = self.results[study_name]

            if study_name == 'positional_encoding':
                ax.plot(study_data['with_pe']['val_losses'], label='With Positional Encoding', linewidth=2)
                ax.plot(study_data['without_pe']['val_losses'], label='Without Positional Encoding', linewidth=2)
                ax.set_title('Positional Encoding Ablation', fontsize=14)
            else:
                for config, data in study_data.items():
                    ax.plot(data['val_losses'], label=f'{config}', linewidth=2)

                if study_name == 'attention_heads':
                    ax.set_title('Attention Heads Ablation', fontsize=14)
                elif study_name == 'layer_depth':
                    ax.set_title('Layer Depth Ablation', fontsize=14)
                elif study_name == 'embedding_dim':
                    ax.set_title('Embedding Dimension Ablation', fontsize=14)
                else:
                    ax.set_title(f'{study_name} Ablation', fontsize=14)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Validation Loss', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(len(studies), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Ablation results saved to {save_path}")

        # 保存详细的数值结果
        results_file = os.path.join(self.config['experiment']['save_dir'], 'ablation_results.txt')
        with open(results_file, 'w', encoding='utf-8') as f:  # 添加UTF-8编码
            f.write("Ablation Study Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Config: {self.config}\n\n")

            for study_name, study_data in self.results.items():
                f.write(f"\n{study_name.upper()}:\n")
                f.write("-" * 30 + "\n")

                if study_name == 'positional_encoding':
                    f.write("With PE - Final Loss: {:.4f}\n".format(study_data['with_pe']['final_val_loss']))
                    f.write("Without PE - Final Loss: {:.4f}\n".format(study_data['without_pe']['final_val_loss']))
                else:
                    for config, data in study_data.items():
                        f.write("{} - Final Loss: {:.4f}\n".format(config, data['final_val_loss']))

    def plot_comparison_with_full_model(self):
        """绘制消融实验与完整模型的对比图"""
        if self.full_model_result is None:
            print("No full model result available. Please run run_full_model() first.")
            return

        ensure_results_dir()
        save_path = os.path.join(self.config['experiment']['save_dir'], 'ablation_comparison.png')

        # 创建对比数据
        comparison_data = {}

        # 添加完整模型结果
        comparison_data['Full Model'] = self.full_model_result['final_val_loss']

        # 添加所有消融实验的最佳结果
        for study_name, study_data in self.results.items():
            if study_name == 'positional_encoding':
                # 对于位置编码，我们比较有位置编码的情况
                comparison_data['With PE'] = study_data['with_pe']['final_val_loss']
                comparison_data['Without PE'] = study_data['without_pe']['final_val_loss']
            else:
                # 对于其他实验，我们找到最佳配置
                best_config = None
                best_loss = float('inf')

                for config, data in study_data.items():
                    if data['final_val_loss'] < best_loss:
                        best_loss = data['final_val_loss']
                        best_config = config

                if best_config is not None:
                    comparison_data[f'{study_name}_{best_config}'] = best_loss

        # 绘制条形图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左侧：最终验证损失对比
        names = list(comparison_data.keys())
        values = list(comparison_data.values())

        # 按值排序
        sorted_indices = np.argsort(values)
        sorted_names = [names[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        bars = ax1.barh(sorted_names, sorted_values, color='skyblue')
        ax1.set_xlabel('Final Validation Loss')
        ax1.set_title('Ablation Study vs Full Model (Final Loss)')
        ax1.grid(True, axis='x', alpha=0.3)

        # 在条形上添加数值标签
        for bar, value in zip(bars, sorted_values):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{value:.4f}', va='center', fontsize=10)

        # 右侧：训练曲线对比
        ax2.plot(self.full_model_result['val_losses'], label='Full Model', linewidth=3, color='red')

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))
        color_idx = 0

        for study_name, study_data in self.results.items():
            if study_name == 'positional_encoding':
                # 只绘制有位置编码的情况
                ax2.plot(study_data['with_pe']['val_losses'],
                         label=f'Positional Encoding',
                         linewidth=2,
                         color=colors[color_idx])
                color_idx += 1
            else:
                # 找到最佳配置
                best_config = None
                best_losses = None
                best_loss = float('inf')

                for config, data in study_data.items():
                    if data['final_val_loss'] < best_loss:
                        best_loss = data['final_val_loss']
                        best_config = config
                        best_losses = data['val_losses']

                if best_config is not None:
                    ax2.plot(best_losses,
                             label=f'{study_name} (best: {best_config})',
                             linewidth=2,
                             color=colors[color_idx])
                    color_idx += 1

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Training Curves Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Ablation comparison saved to {save_path}")

        # 保存对比数据为JSON
        comparison_json = os.path.join(self.config['experiment']['save_dir'], 'ablation_comparison.json')
        with open(comparison_json, 'w', encoding='utf-8') as f:  # 添加UTF-8编码
            json.dump(comparison_data, f, indent=2)

        print(f"Comparison data saved to {comparison_json}")

        return comparison_data


def main():
    # 加载配置
    config = load_config()

    set_seed(int(config['experiment']['seed']))
    device = get_device(config)

    ablation_study = AblationStudy(config)

    # 首先运行完整模型作为基准
    ablation_study.run_full_model()

    # 运行消融实验
    ablation_study.run_positional_encoding_ablation()
    ablation_study.run_attention_heads_ablation()
    ablation_study.run_layer_depth_ablation()

    # 绘制结果
    ablation_study.plot_results()

    # 绘制与完整模型的对比
    comparison_data = ablation_study.plot_comparison_with_full_model()

    print("Ablation study completed! All results saved to results/ folder")

    # 打印最佳配置建议
    print("\n" + "=" * 50)
    print("BEST CONFIGURATION RECOMMENDATIONS")
    print("=" * 50)

    if comparison_data:
        # 找到最佳配置
        best_config = min(comparison_data, key=comparison_data.get)
        best_loss = comparison_data[best_config]
        full_model_loss = comparison_data['Full Model']

        print(f"Best configuration: {best_config} (Loss: {best_loss:.4f})")
        print(f"Full model loss: {full_model_loss:.4f}")

        improvement = ((full_model_loss - best_loss) / full_model_loss) * 100
        if improvement > 0:
            print(f"Improvement over full model: {improvement:.2f}%")
        else:
            print(f"Full model performed better by: {-improvement:.2f}%")

    # 分析每个消融实验的最佳配置
    for study_name, study_data in ablation_study.results.items():
        if study_name == 'positional_encoding':
            with_pe_loss = study_data['with_pe']['final_val_loss']
            without_pe_loss = study_data['without_pe']['final_val_loss']

            if with_pe_loss < without_pe_loss:
                print(
                    f"RECOMMENDED Positional Encoding: USE positional encoding (loss: {with_pe_loss:.4f} vs {without_pe_loss:.4f})")
            else:
                print(
                    f"NOT RECOMMENDED Positional Encoding: AVOID positional encoding (loss: {without_pe_loss:.4f} vs {with_pe_loss:.4f})")
        else:
            best_config = None
            best_loss = float('inf')

            for config, data in study_data.items():
                if data['final_val_loss'] < best_loss:
                    best_loss = data['final_val_loss']
                    best_config = config

            if best_config is not None:
                print(
                    f"BEST {study_name.replace('_', ' ').title()}: Best configuration is {best_config} (loss: {best_loss:.4f})")


if __name__ == '__main__':
    main()