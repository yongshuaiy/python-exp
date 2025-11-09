#!/usr/bin/env python3
"""
Transformer from Scratch - 主程序入口
直接运行此文件开始训练和消融实验
"""

import sys
import os

# 添加当前目录到Python路径，确保可以导入src模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 50)
    print("Transformer-exp-25125355")
    print("=" * 50)



    # 运行训练
    print("\n开始训练主模型")
    print("-" * 30)
    from src.train import main as train_main
    train_main()

    # 运行消融实验
    print("\n开始消融实验")
    print("-" * 30)
    from src.ablation import main as ablation_main
    ablation_main()

    # 运行对比分析
    print("\n开始对比分析")
    print("-" * 30)
    from compare_ablation import main as compare_main
    compare_main()

    print("\n所有实验完成！结果保存在 results/ 文件夹中")


if __name__ == '__main__':
    main()