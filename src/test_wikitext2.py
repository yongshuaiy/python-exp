# verify_wikitext2.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_wikitext2_offline


def verify_data():
    """验证WikiText-2数据"""
    print("=== 验证WikiText-2数据 ===")

    try:
        train_loader, val_loader, vocab = load_wikitext2_offline(
            seq_len=64,
            batch_size=4,
            use_sample_data=False  # 强制使用真实数据
        )

        print("✅ WikiText-2数据加载成功!")
        print(f"词汇表大小: {len(vocab)}")

        # 检查训练数据
        train_samples = 0
        for i, (data, target) in enumerate(train_loader):
            train_samples += data.shape[0]
            if i == 0:  # 只检查第一个批次
                print(f"数据形状: {data.shape}")
                print(f"目标形状: {target.shape}")

                # 解码示例
                sample_tokens = data[0][:15].tolist()
                sample_text = " ".join([vocab.itos.get(t, f"<unk_{t}>") for t in sample_tokens])
                print(f"样本文本: {sample_text}")

                # 检查数据范围
                print(f"Token范围: {data.min().item()} - {data.max().item()}")
                print(f"词汇表范围: 0 - {len(vocab) - 1}")

        print(f"训练批次数量: {i + 1}")
        print(f"总训练样本: {train_samples}")

        # 检查验证数据
        val_samples = 0
        for i, (data, target) in enumerate(val_loader):
            val_samples += data.shape[0]

        print(f"验证批次数量: {i + 1}")
        print(f"总验证样本: {val_samples}")

        return True

    except Exception as e:
        print(f"❌ 数据验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = verify_data()
    if success:
        print("\n🎉 WikiText-2数据验证成功!")
    else:
        print("\n💥 WikiText-2数据验证失败!")