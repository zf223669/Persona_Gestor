import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch

def interactive_attention_heatmap(attn_weights, title="Interactive Attention Heatmap"):
    """
    创建交互式注意力热力图
    """
    if isinstance(attn_weights, torch.Tensor):
        attn_np = attn_weights.detach().cpu().numpy()

    # 取第一个样本的平均注意力
    if len(attn_np.shape) > 2:
        avg_attn = np.mean(attn_np[0], axis=0) if len(attn_np.shape) == 4 else np.mean(attn_np, axis=0)
    else:
        avg_attn = attn_np

    fig = go.Figure(data=go.Heatmap(z=avg_attn, colorscale='Viridis'))
    fig.update_layout(
        title=title,
        xaxis_title='Key Positions',
        yaxis_title='Query Positions'
    )
    fig.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os


def save_attention_heatmaps(attn_weights, save_dir="attention_plots", prefix="attn_heatmap", show_grid=False):
    """
    使用seaborn绘制注意力热力图并保存为文件

    Args:
        attn_weights: 注意力权重张量
        save_dir: 保存目录
        prefix: 文件名前缀
        show_grid: 是否显示网格线
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(attn_weights, torch.Tensor):
        attn_np = attn_weights.detach().cpu().numpy()

    # 获取文件序号
    existing_files = [f for f in os.listdir(save_dir) if f.startswith(prefix)]
    start_idx = len(existing_files)

    # 设置网格线参数
    linewidths = 0.05 if show_grid else 0
    linecolor = 'white' if show_grid else 'none'

    # 处理不同维度的注意力权重
    if len(attn_np.shape) == 4:  # (batch, heads, seq_len, seq_len)
        batch_size, num_heads, seq_len, _ = attn_np.shape
        file_idx = start_idx

        for b in range(batch_size):
            for h in range(num_heads):
                attn_map = attn_np[b, h]  # 提取单个头的注意力

                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    attn_map,
                    annot=False,  # 是否显示数值
                    cmap='viridis',
                    cbar=True,
                    square=True,
                    linewidths=linewidths,  # 网格线宽度
                    linecolor=linecolor  # 网格线颜色
                )

                plt.title(f'Attention Heatmap - Batch {b}, Head {h}')
                plt.xlabel('Key Positions')
                plt.ylabel('Query Positions')

                # 保存为文件
                filename = f"{prefix}_{file_idx:03d}.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()  # 关闭图形释放内存

                file_idx += 1

    elif len(attn_np.shape) == 3:  # (heads, seq_len, seq_len)
        num_heads, seq_len, _ = attn_np.shape

        for h in range(num_heads):
            attn_map = attn_np[h]

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                attn_map,
                annot=False,
                cmap='viridis',
                cbar=True,
                square=True,
                linewidths=linewidths,
                linecolor=linecolor
            )

            plt.title(f'Attention Heatmap - Head {h}')
            plt.xlabel('Key Positions')
            plt.ylabel('Query Positions')

            filename = f"{prefix}_{start_idx + h:03d}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

    elif len(attn_np.shape) == 2:  # (seq_len, seq_len)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attn_np,
            annot=False,
            cmap='viridis',
            cbar=True,
            square=True,
            linewidths=linewidths,
            linecolor=linecolor
        )

        plt.title('Attention Heatmap')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')

        filename = f"{prefix}_{start_idx:03d}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"已保存注意力热力图到 {save_dir} 目录")

def visualize_and_save_attention(attn_weights):
    save_attention_heatmaps(attn_weights, save_dir="attn_visualizations", prefix="attn")

