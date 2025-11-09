import yaml
import torch

"""加载配置文件并确保数值类型正确"""
def load_config(config_path="./configs/config.yaml"):
    print(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 确保数值参数是正确的类型
    _ensure_numeric_types(config)
    return config


def _ensure_numeric_types(config):
    """递归确保配置中的数值类型正确"""
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict):
                _ensure_numeric_types(value)
            elif isinstance(value, str):
                # 尝试将字符串转换为数值
                if value.replace('.', '').replace('-', '').isdigit():
                    if '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                elif value.lower() in ['true', 'false']:
                    config[key] = value.lower() == 'true'
    elif isinstance(config, list):
        for i, item in enumerate(config):
            if isinstance(item, (dict, list)):
                _ensure_numeric_types(item)
            elif isinstance(item, str):
                if item.replace('.', '').replace('-', '').isdigit():
                    if '.' in item:
                        config[i] = float(item)
                    else:
                        config[i] = int(item)


def get_device(config):
    """获取设备"""
    device_str = config.get('experiment', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)