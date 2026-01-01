import pandas as pd
import matplotlib.pyplot as plt
import sys

try:
    # 读取 CSV 数据
    df = pd.read_csv('performance_results.csv', names=['width', 'height', 'kernel_type', 'ksize','iterations', 'avg_time_ms', 'throughput_MPs'])
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图表1：不同图像大小的性能
    size_data = df[df['ksize'] == 3].sort_values('width')
    axes[0, 0].plot(size_data['width'] * size_data['height'] / 1e6, size_data['avg_time_ms'], 'o-')
    axes[0, 0].set_xlabel('图像大小 (百万像素)')
    axes[0, 0].set_ylabel('平均时间 (ms)')
    axes[0, 0].set_title('不同图像大小的性能')
    axes[0, 0].grid(True)
    
    # 图表2：不同内核大小的性能
    kernel_data = df[(df['width'] == 512) & (df['height'] == 512)]
    axes[0, 1].plot(kernel_data['ksize'], kernel_data['avg_time_ms'], 's-')
    axes[0, 1].set_xlabel('内核大小')
    axes[0, 1].set_ylabel('平均时间 (ms)')
    axes[0, 1].set_title('不同内核大小的性能 (512x512)')
    axes[0, 1].grid(True)
    
    # 图表3：吞吐量对比
    axes[1, 0].bar(range(len(kernel_data)), kernel_data['throughput_MPs'])
    axes[1, 0].set_xlabel('测试用例')
    axes[1, 0].set_ylabel('吞吐量 (MP/s)')
    axes[1, 0].set_title('吞吐量对比')
    axes[1, 0].set_xticks(range(len(kernel_data)))
    axes[1, 0].set_xticklabels([f"K{t}" for t in kernel_data['kernel_type']])
    axes[1, 0].grid(True, axis='y')
    
    # 图表4：性能汇总表
    axes[1, 1].axis('off')
    summary_text = "性能测试汇总\n\n"
    for idx, row in df.iterrows():
        summary_text += f"Test {idx+1}: {row['width']}x{row['height']}, " \
                       f"K{row['kernel_type']}, Size {row['ksize']}\n" \
                       f"  Time: {row['avg_time_ms']:.2f} ms, " \
                       f"Throughput: {row['throughput_MPs']:.2f} MP/s\n\n"
    axes[1, 1].text(0, 1, summary_text, fontsize=8, 
                    verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig('performance_chart.png', dpi=150)
    print("性能图表已保存为 performance_chart.png")
    plt.show()
    
except FileNotFoundError:
    print("错误：未找到 performance_results.csv 文件")
except Exception as e:
    print(f"错误：{e}")