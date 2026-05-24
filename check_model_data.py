#!/usr/bin/env python3
"""
查询 MMSA 模型需要的数据文件类型
"""
import json
import sys
from pathlib import Path


def find_config_file():
    """查找 MMSA 配置文件"""
    try:
        import MMSA
        mmsa_path = Path(MMSA.__file__).parent
        config_path = mmsa_path / "config" / "config_regression.json"
        if config_path.exists():
            return config_path
    except ImportError:
        pass
    
    # 备用路径
    venv_path = Path(__file__).parent / ".venv" / "lib"
    for config_path in venv_path.rglob("config_regression.json"):
        return config_path
    
    return None


def check_model_alignment(model_name=None):
    """检查模型的对齐需求"""
    config_path = find_config_file()
    
    if not config_path:
        print("❌ 找不到 MMSA 配置文件")
        return
    
    with config_path.open("r") as f:
        config = json.load(f)
    
    if model_name:
        # 查询单个模型
        if model_name not in config:
            print(f"❌ 模型 '{model_name}' 不存在")
            print(f"\n可用模型: {', '.join(sorted(config.keys()))}")
            return
        
        model_config = config[model_name]["commonParams"]
        need_aligned = model_config.get("need_data_aligned", False)
        need_model_aligned = model_config.get("need_model_aligned", False)
        
        print(f"\n模型: {model_name}")
        print(f"  need_data_aligned:  {need_aligned}")
        print(f"  need_model_aligned: {need_model_aligned}")
        print(f"\n推荐数据文件:")
        if need_aligned:
            print(f"  MOSI:  aligned_50.pkl")
            print(f"  MOSEI: mosei_senti_data.pkl")
        else:
            print(f"  MOSI:  unaligned_50.pkl")
            print(f"  MOSEI: mosei_senti_data_noalign.pkl")
    else:
        # 列出所有模型
        print("\n所有模型的对齐需求:\n")
        print(f"{'模型':<20} {'need_data_aligned':<20} {'推荐文件'}")
        print("-" * 70)
        
        for model in sorted(config.keys()):
            model_config = config[model]["commonParams"]
            need_aligned = model_config.get("need_data_aligned", False)
            file_type = "aligned" if need_aligned else "unaligned"
            print(f"{model:<20} {str(need_aligned):<20} {file_type}")


def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        check_model_alignment(model_name)
    else:
        print("用法:")
        print("  查询单个模型: python check_model_data.py <model_name>")
        print("  列出所有模型: python check_model_data.py")
        print("\n示例:")
        print("  python check_model_data.py ef_lstm")
        print("  python check_model_data.py lmf")
        print()
        check_model_alignment()


if __name__ == "__main__":
    main()
