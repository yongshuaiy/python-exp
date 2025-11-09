import torch
from torch.utils.data import Dataset, DataLoader
import os
import requests
import zipfile
from collections import Counter
import re
import tarfile


class LMDataset(Dataset):
    """语言建模数据集"""

    def __init__(self, data, seq_len=128):
        self.data = data
        self.seq_len = int(seq_len)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x.clone().detach(), y.clone().detach()


def create_simple_dataset(vocab_size=1000, num_samples=10000, seq_len=128, batch_size=32):
    """创建简单的合成数据集"""
    vocab_size = int(vocab_size)
    num_samples = int(num_samples)
    seq_len = int(seq_len)
    batch_size = int(batch_size)

    data = torch.randint(1, vocab_size, (num_samples,))

    dataset = LMDataset(data, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class SimpleVocab:
        def __init__(self, size):
            self.size = int(size)
            self.itos = {i: f"word_{i}" for i in range(size)}
            self.itos[0] = "<pad>"
            self.stoi = {v: k for k, v in self.itos.items()}

        def __len__(self):
            return self.size

    vocab = SimpleVocab(vocab_size)

    return loader, loader, vocab


def download_wikitext2():
    """下载WikiText-2数据集 - 多个备选源"""
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)

    # 多个备选下载源
    download_sources = [
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/",
    ]

    zip_path = os.path.join(data_dir, "wikitext-2-v1.zip")
    extract_path = os.path.join(data_dir, "wikitext-2")

    # 如果已经解压，直接返回路径
    if os.path.exists(extract_path):
        return extract_path

    # 如果ZIP文件不存在，尝试下载
    if not os.path.exists(zip_path):
        print("尝试下载WikiText-2数据集...")

        for url in download_sources:
            try:
                print(f"尝试从 {url} 下载...")

                if url.endswith('.zip'):
                    # 下载ZIP文件
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()

                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0

                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    print(f"下载进度: {percent:.1f}%", end='\r')

                    print("\n下载完成!")
                    break

            except Exception as e:
                print(f"从 {url} 下载失败: {e}")
                continue
        else:
            print("所有下载源都失败了")
            return None

    # 尝试解压文件
    print("尝试解压数据集...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 检查是否是有效的ZIP文件
            file_list = zip_ref.namelist()
            if not file_list:
                raise ValueError("ZIP文件为空")

            zip_ref.extractall(data_dir)
        print("解压完成!")
        return extract_path

    except zipfile.BadZipFile:
        print("ZIP文件损坏，尝试其他方式...")
        # 如果ZIP文件损坏，尝试直接创建目录结构
        return create_wikitext2_manually()
    except Exception as e:
        print(f"解压失败: {e}")
        return None


def create_wikitext2_manually():
    """手动创建WikiText-2目录结构（备选方案）"""
    data_dir = "../data"
    wikitext_dir = os.path.join(data_dir, "wikitext-2")
    os.makedirs(wikitext_dir, exist_ok=True)

    # 创建空文件作为占位符
    files = ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']
    for file in files:
        file_path = os.path.join(wikitext_dir, file)
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# WikiText-2数据集文件\n")
                f.write("# 请手动下载并替换此文件\n")
                f.write("# 下载地址: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip\n")

    print("已创建WikiText-2目录结构，请手动下载数据集文件")
    return wikitext_dir


# def simple_tokenizer(text):
#     """简单分词器"""
#     # 基本的分词：按空格分割，并清理标点符号
#     text = re.sub(r'[^\w\s]', ' ', text)
#     tokens = text.lower().split()
#     return [token for token in tokens if token.strip()]
def simple_tokenizer(text):
    """改进的分词器"""
    # 更细致的文本清理
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格
    tokens = text.lower().strip().split()
    return [token for token in tokens if len(token) > 1]  # 过滤掉单字符token

def create_sample_wikitext2_data():
    """创建示例WikiText-2格式的数据（用于测试）"""
    data_dir = "../data"
    wikitext_dir = os.path.join(data_dir, "wikitext-2")
    os.makedirs(wikitext_dir, exist_ok=True)

    # 创建示例数据
    sample_data = {
        'wiki.train.tokens': [
            "= Example Article =",
            "This is a sample training text for testing the transformer model .",
            "It contains multiple sentences to demonstrate the data loading process .",
            "The model should learn to predict the next word in a sequence .",
            "This is important for language modeling tasks ."
        ],
        'wiki.valid.tokens': [
            "= Validation Text =",
            "This is a sample validation text .",
            "It is used to evaluate the model performance .",
            "The validation loss indicates how well the model generalizes ."
        ],
        'wiki.test.tokens': [
            "= Test Text =",
            "This is a sample test text .",
            "It provides the final evaluation of the model .",
            "Good performance on test data shows the model works well ."
        ]
    }

    for filename, lines in sample_data.items():
        file_path = os.path.join(wikitext_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

    print("已创建示例WikiText-2数据")
    return wikitext_dir


def load_wikitext2_offline(seq_len=128, batch_size=32, use_sample_data=False,
                           max_train_tokens=50000, max_val_tokens=10000, vocab_size_limit=5000):
    # """完全离线的WikiText-2加载 - 支持数据量限制"""
    # print("使用离线方式加载WikiText-2...")
    # print(f"数据限制: 训练token ≤ {max_train_tokens}, 验证token ≤ {max_val_tokens}, 词汇表 ≤ {vocab_size_limit}")

    # 尝试下载或使用本地数据
    # data_path = download_wikitext2()
    data_path = "./data/wikitext-2"

    if data_path is None and use_sample_data:
        print("使用示例数据...")
        data_path = create_sample_wikitext2_data()

    if data_path is None:
        # 尝试直接使用可能存在的本地文件
        possible_paths = [
            "data/wikitext-2",
            "wikitext-2",
            "../data/wikitext-2",
            "./wikitext-2"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        else:
            print("找不到WikiText-2数据集文件")
            raise FileNotFoundError("WikiText-2数据集不存在")

    # 定义文件路径
    train_file = os.path.join(data_path, "wiki.train.tokens")
    valid_file = os.path.join(data_path, "wiki.valid.tokens")
    test_file = os.path.join(data_path, "wiki.test.tokens")

    # 检查文件是否存在且非空
    files_exist = all(os.path.exists(f) for f in [train_file, valid_file, test_file])
    files_non_empty = all(os.path.getsize(f) > 100 for f in [train_file, valid_file, test_file])  # 大于100字节

    if not files_exist or not files_non_empty:
        if use_sample_data:
            print("数据集文件不存在或为空，使用示例数据...")
            data_path = create_sample_wikitext2_data()
            train_file = os.path.join(data_path, "wiki.train.tokens")
            valid_file = os.path.join(data_path, "wiki.valid.tokens")
            test_file = os.path.join(data_path, "wiki.test.tokens")
        else:
            print(f"数据集文件不完整，请检查 {data_path} 目录")
            raise FileNotFoundError("WikiText-2数据集文件不完整")

    print(f"从 {data_path} 加载数据...")

    def read_file(file_path):
        """读取文件并返回文本行"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('=')]
            return lines
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('=')]
            return lines

    # 读取数据
    # print("读取训练文件...")
    train_lines = read_file(train_file)
    # print("读取验证文件...")
    valid_lines = read_file(valid_file)
    # print("读取测试文件...")
    test_lines = read_file(test_file)

    print(f"训练集: {len(train_lines)} 行")
    print(f"验证集: {len(valid_lines)} 行")
    print(f"测试集: {len(test_lines)} 行")

    # 如果数据太少，使用示例数据
    if len(train_lines) < 10 and use_sample_data:
        print("实际数据太少，使用示例数据...")
        data_path = create_sample_wikitext2_data()
        train_lines = read_file(os.path.join(data_path, "wiki.train.tokens"))
        valid_lines = read_file(os.path.join(data_path, "wiki.valid.tokens"))
        test_lines = read_file(os.path.join(data_path, "wiki.test.tokens"))

    # 构建词汇表（仅使用训练集）
    print("构建词汇表...")
    all_tokens = []
    token_limit = 100000  # 只使用前10万个token来构建词汇表，加快处理速度

    for i, line in enumerate(train_lines):
        if len(all_tokens) >= token_limit:
            break
        tokens = simple_tokenizer(line)
        all_tokens.extend(tokens)

    # 创建词汇表，限制大小
    token_counts = Counter(all_tokens)
    # vocab_tokens = ['<pad>', '<unk>'] + [token for token, count in token_counts.most_common(vocab_size_limit - 2)]
    # 在构建词汇表时添加更多过滤
    min_freq = 3  # 从2增加到3，只保留出现3次以上的词
    vocab_tokens = ['<pad>', '<unk>'] + [token for token, count in token_counts.most_common()
                                         if count >= min_freq and len(token) > 1][:vocab_size_limit - 2]

    # 如果实际词汇表小于限制，使用实际大小
    actual_vocab_size = len(vocab_tokens)
    print(f"实际词汇表大小: {actual_vocab_size}")

    class SimpleVocab:
        def __init__(self, tokens):
            self.itos = {i: token for i, token in enumerate(tokens)}
            self.stoi = {token: i for i, token in enumerate(tokens)}
            self.size = len(tokens)

        def __len__(self):
            return self.size

        def __getitem__(self, token):
            return self.stoi.get(token, 1)  # 1是<unk>的索引

        def get_itos(self):
            return self.itos

        def get_stoi(self):
            return self.stoi

    vocab = SimpleVocab(vocab_tokens)
    print(f"词汇表构建完成，大小: {len(vocab)}")

    # 处理数据：将文本转换为token IDs，并限制数据量
    def text_to_ids(lines, max_tokens=None):
        data = []
        for line in lines:
            if max_tokens and len(data) >= max_tokens:
                break
            tokens = simple_tokenizer(line)
            token_ids = [vocab[token] for token in tokens]
            # 如果添加这行会超过限制，则跳过剩余部分
            if max_tokens and len(data) + len(token_ids) > max_tokens:
                remaining = max_tokens - len(data)
                data.extend(token_ids[:remaining])
                break
            else:
                data.extend(token_ids)
        return torch.tensor(data, dtype=torch.long)

    # print("处理训练数据...")
    train_data = text_to_ids(train_lines, max_train_tokens)
    # print("处理验证数据...")
    val_data = text_to_ids(valid_lines, max_val_tokens)

    print(f"训练数据: {len(train_data)} 个token")
    print(f"验证数据: {len(val_data)} 个token")

    # 创建数据集
    train_dataset = LMDataset(train_data, seq_len)
    val_dataset = LMDataset(val_data, seq_len)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("WikiText-2数据加载完成!")
    return train_loader, val_loader, vocab




def load_data(config):
    """根据配置加载数据 - 支持数据量限制"""
    data_config = config['data']
    dataset_name = data_config['dataset']
    batch_size = int(data_config['batch_size'])
    seq_len = int(data_config['seq_len'])
    vocab_size = int(data_config['vocab_size'])

    # 新增的数据限制参数
    max_train_tokens = data_config.get('max_train_tokens', 50000)
    max_val_tokens = data_config.get('max_val_tokens', 10000)

    print(f"Loading dataset: {dataset_name}")
    print(f"Data limits - Train: {max_train_tokens} tokens, Val: {max_val_tokens} tokens")

    if dataset_name == 'wikitext2':
        try:
            print("正在加载限制大小的WikiText-2数据...")
            train_loader, val_loader, vocab = load_wikitext2_offline(
                seq_len=seq_len,
                batch_size=batch_size,
                use_sample_data=False,
                max_train_tokens=max_train_tokens,
                max_val_tokens=max_val_tokens,
                vocab_size_limit=vocab_size
            )

            actual_vocab_size = len(vocab)
            print(f"✅ WikiText-2数据加载成功!")
            print(f"   词汇表大小: {actual_vocab_size}")
            print(f"   训练token数量: {sum(len(batch[0]) for batch in train_loader)}")
            print(f"   验证token数量: {sum(len(batch[0]) for batch in val_loader)}")

            return train_loader, val_loader, vocab

        except Exception as e:
            print(f"❌ 真实WikiText-2数据加载失败: {e}")
            print("回退到合成数据...")
            num_samples = int(data_config.get('num_samples', 1000))
            return create_simple_dataset(
                vocab_size=vocab_size,
                num_samples=num_samples,
                seq_len=seq_len,
                batch_size=batch_size
            )
    else:
        num_samples = int(data_config.get('num_samples', 1000))
        return create_simple_dataset(
            vocab_size=vocab_size,
            num_samples=num_samples,
            seq_len=seq_len,
            batch_size=batch_size
        )

# def load_data(config):
#     """根据配置加载数据 - 优先使用真实WikiText-2数据"""
#     data_config = config['data']
#     dataset_name = data_config['dataset']
#     batch_size = int(data_config['batch_size'])
#     seq_len = int(data_config['seq_len'])
#     vocab_size = int(data_config['vocab_size'])
#     num_samples = int(data_config['num_samples'])
#
#     print(f"Loading dataset: {dataset_name}")
#
#     if dataset_name == 'wikitext2':
#         try:
#             print("正在加载真实的WikiText-2数据...")
#             train_loader, val_loader, vocab = load_wikitext2_offline(
#                 seq_len=seq_len,
#                 batch_size=batch_size,
#                 use_sample_data=False  # 强制使用真实数据
#             )
#
#             # 验证词汇表大小是否匹配配置
#             actual_vocab_size = len(vocab)
#             print(f"✅ 真实WikiText-2数据加载成功!")
#             print(f"   词汇表大小: {actual_vocab_size}")
#             print(f"   配置词汇表大小: {vocab_size}")
#
#             # 统计训练样本数量
#             train_count = sum(1 for _ in train_loader)
#             val_count = sum(1 for _ in val_loader)
#
#             print(f"   训练批次: {train_count}")
#             print(f"   验证批次: {val_count}")
#
#             return train_loader, val_loader, vocab
#
#         except Exception as e:
#             print(f"❌ 真实WikiText-2数据加载失败: {e}")
#             print("回退到合成数据...")
#             return create_simple_dataset(
#                 vocab_size=vocab_size,
#                 num_samples=num_samples,
#                 seq_len=seq_len,
#                 batch_size=batch_size
#             )
#     else:
#         # 使用合成数据
#         return create_simple_dataset(
#             vocab_size=vocab_size,
#             num_samples=num_samples,
#             seq_len=seq_len,
#             batch_size=batch_size
#         )