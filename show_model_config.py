#!/usr/bin/env python3
"""
查看 MMSA 模型在特定数据集上的完整配置
"""
import json
import sys
from MMSA import get_config_regression


def show_config(model_name, dataset_name='mosi'):
    """显示模型配置"""
    try:
        config = get_config_regression(model_name, dataset_name)
        
        print(f"\n{'='*70}")
        print(f"模型: {model_name}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*70}\n")
        
        # 分类显示
        print("【数据相关】")
        data_keys = ['need_data_aligned', 'need_model_aligned', 'need_normalized', 
                     'use_bert', 'use_finetune']
        for key in data_keys:
            if key in config:
                print(f"  {key:<25} {config[key]}")
        
        print("\n【训练相关】")
        train_keys = ['batch_size', 'learning_rate', 'weight_decay', 'grad_clip', 
                      'early_stop', 'patience']
        for key in train_keys:
            if key in config:
                print(f"  {key:<25} {config[key]}")
        
        print("\n【模型结构】")
        model_keys = ['hidden_dims', 'rank', 'dropout', 'dropouts', 'num_layers',
                      'text_out', 'audio_out', 'video_out', 'post_fusion_dim',
                      'nheads', 'nlevels', 'dst_feature_dim_nheads']
        for key in model_keys:
            if key in config:
                print(f"  {key:<25} {config[key]}")
        
        print("\n【其他参数】")
        shown_keys = set(data_keys + train_keys + model_keys)
        for key, value in sorted(config.items()):
            if key not in shown_keys:
                print(f"  {key:<25} {value}")
        
        print(f"\n{'='*70}\n")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


def main():
    if len(sys.argv) < 2:
        print("用法: python show_model_config.py <model_name> [dataset_name]")
        print("\n示例:")
        print("  python show_model_config.py lmf")
        print("  python show_model_config.py lmf mosi")
        print("  python show_model_config.py mult mosei")
        print("\n常用模型: ef_lstm, lf_dnn, lmf, mult, self_mm")
        print("常用数据集: mosi, mosei, sims")
        return
    
    model_name = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'mosi'
    
    show_config(model_name, dataset_name)


if __name__ == "__main__":
    main()
