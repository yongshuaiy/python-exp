#!/usr/bin/env python3
"""
对比消融实验与完整模型结果的脚本
"""

import sys
import os
import json

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ablation import AblationStudy
from src.config import load_config


def main():
    print("=" * 50)
    print("Ablation Study Comparison")
    print("=" * 50)

    # 加载配置
    config = load_config()

    ablation_study = AblationStudy(config)

    # 运行完整模型和消融实验
    ablation_study.run_full_model()
    ablation_study.run_positional_encoding_ablation()
    ablation_study.run_attention_heads_ablation()
    ablation_study.run_layer_depth_ablation()

    # 绘制对比图
    comparison_data = ablation_study.plot_comparison_with_full_model()

    # 生成配置建议报告
    generate_config_report(ablation_study, comparison_data)


def generate_config_report(ablation_study, comparison_data):
    """生成配置建议报告"""
    report_path = os.path.join('results', 'configuration_recommendations.txt')

    with open(report_path, 'w', encoding='utf-8') as f:  # 添加UTF-8编码
        f.write("=" * 60 + "\n")
        f.write("TRANSFORMER CONFIGURATION RECOMMENDATIONS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Based on ablation study results, here are the optimal configurations:\n\n")

        # 总体最佳配置
        if comparison_data:
            best_config = min(comparison_data, key=comparison_data.get)
            best_loss = comparison_data[best_config]
            full_model_loss = comparison_data['Full Model']

            f.write("OVERALL BEST CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Configuration: {best_config}\n")
            f.write(f"Validation Loss: {best_loss:.4f}\n")
            f.write(f"Full Model Loss: {full_model_loss:.4f}\n")

            improvement = ((full_model_loss - best_loss) / full_model_loss) * 100
            if improvement > 0:
                f.write(f"Improvement: {improvement:.2f}%\n")
            else:
                f.write(f"Full model performed better by: {-improvement:.2f}%\n")

            f.write("\n")

        # 各组件最佳配置
        f.write("COMPONENT-WISE RECOMMENDATIONS:\n")
        f.write("-" * 30 + "\n")

        for study_name, study_data in ablation_study.results.items():
            if study_name == 'positional_encoding':
                with_pe_loss = study_data['with_pe']['final_val_loss']
                without_pe_loss = study_data['without_pe']['final_val_loss']

                if with_pe_loss < without_pe_loss:
                    f.write("RECOMMENDED Positional Encoding: USE (improves performance)\n")
                    f.write(f"  - With PE: {with_pe_loss:.4f}\n")
                    f.write(f"  - Without PE: {without_pe_loss:.4f}\n")
                else:
                    f.write("NOT RECOMMENDED Positional Encoding: AVOID (degrades performance)\n")
                    f.write(f"  - With PE: {with_pe_loss:.4f}\n")
                    f.write(f"  - Without PE: {without_pe_loss:.4f}\n")
            else:
                best_config = None
                best_loss = float('inf')
                all_configs = []

                for config, data in study_data.items():
                    all_configs.append((config, data['final_val_loss']))
                    if data['final_val_loss'] < best_loss:
                        best_loss = data['final_val_loss']
                        best_config = config

                if best_config is not None:
                    f.write(f"BEST {study_name.replace('_', ' ').title()}: {best_config}\n")
                    f.write(f"  - Best loss: {best_loss:.4f}\n")

                    # 显示所有配置的损失
                    all_configs.sort(key=lambda x: x[1])
                    for config, loss in all_configs:
                        f.write(f"  - {config}: {loss:.4f}\n")

            f.write("\n")

        # 推荐配置
        f.write("RECOMMENDED CONFIGURATION:\n")
        f.write("-" * 30 + "\n")

        recommended_config = {}

        for study_name, study_data in ablation_study.results.items():
            if study_name == 'positional_encoding':
                with_pe_loss = study_data['with_pe']['final_val_loss']
                without_pe_loss = study_data['without_pe']['final_val_loss']

                if with_pe_loss < without_pe_loss:
                    recommended_config['use_positional_encoding'] = True
                else:
                    recommended_config['use_positional_encoding'] = False
            else:
                best_config = None
                best_loss = float('inf')

                for config, data in study_data.items():
                    if data['final_val_loss'] < best_loss:
                        best_loss = data['final_val_loss']
                        best_config = config

                if best_config is not None:
                    recommended_config[study_name] = best_config

        for key, value in recommended_config.items():
            f.write(f"{key}: {value}\n")

    print(f"Configuration recommendations saved to {report_path}")


if __name__ == '__main__':
    main()