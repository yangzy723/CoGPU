import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# ========== 1. 全局出版级样式设置 ==========
# 统一使用 Serif 字体，匹配 ACM/IEEE 论文排版
plt.rcParams['font.family'] = 'serif'
# 设置更高分辨率，确保 PDF 矢量图的清晰度
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['axes.linewidth'] = 1.0 # 坐标轴线宽

# ========== 2. 数据定义 ==========
categories = ['A', 'B', '1', '2']
systems = ['Temporal', 'MPS', 'MIG', 'Orion', 'LithOS', 'CoGPU']

throughput_data = {
    'Temporal': [0.44, 0.40, 0.42, 0.43],
    'MPS':      [0.50, 0.54, 0.55, 0.54],
    'MIG':      [0.50, 0.50, 0.53, 0.50],
    'Orion':    [0.50, 0.55, 0.53, 0.51],
    'LithOS':   [0.53, 0.53, 0.58, 0.54],
    'CoGPU':    [0.55, 0.55, 0.60, 0.60]
}

latency_data = {
    'Temporal': [7.2, 8.9, 9.0, 9.6],
    'MPS':      [9.0, 10.2, 9.5, 10.6],
    'MIG':      [10.0, 12.0, 10.0, 12.0],
    'Orion':    [12.0, 15.0, 15.0, 19.0],
    'LithOS':   [5.3, 6.5, 7.6, 8.2],
    'CoGPU':    [4.5, 5.9, 7.2, 8.0]
}

# 颜色和纹理配置（保留了柔和高级的配色方案）
colors = ['#d9d9d9', '#c4b5db', '#f5c687', '#9dc3e6', '#a9d18e', '#f44336']
hatches = ['', '///', '\\\\\\', 'xxx', '...', '']

x = np.arange(len(categories))
width = 0.12

# 调整画布大小，使其更适合双栏横跨布局（更扁平紧凑）
fig, axes = plt.subplots(2, 1, figsize=(10, 6.8)) 

# ========== 3. 核心绘制逻辑 ==========
def draw_bars(ax, data_dict, is_throughput):
    for i, sys in enumerate(systems):
        pos = x + (i - 2.5) * width 
        
        if sys == 'LithOS':
            # LithOS：幽灵虚线框，强调没有 Semantic Determinism 保证
            bars = ax.bar(pos, data_dict[sys], width, label=sys,
                          facecolor='none',          
                          edgecolor=colors[i],       
                          linestyle='--',            
                          linewidth=1.5,             
                          hatch=hatches[i])          
        else:
            # 常规系统与 CoGPU
            bars = ax.bar(pos, data_dict[sys], width, label=sys,
                          color=colors[i], 
                          edgecolor='black', # 锐利的黑色边框
                          linestyle='-', 
                          linewidth=0.75, 
                          hatch=hatches[i])
        
        # 数值标签添加逻辑
        for bar in bars:
            height = bar.get_height()
            label_text = f'{height:.2f}' if is_throughput else f'{height:.1f}'
            
            if sys == 'LithOS':
                label_text += '*'
                
            # 调整：字号调大为10.5，颜色略加深以保证可读性
            ax.annotate(label_text,
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), # 距离柱子顶部 4 个像素
                         textcoords="offset points",
                         ha='center', va='bottom', rotation=90,
                         fontweight='bold', fontsize=10.5, color='#222222')

# 执行绘制
draw_bars(axes[0], throughput_data, is_throughput=True)
draw_bars(axes[1], latency_data, is_throughput=False)

# ========== 4. 坐标轴与背景美化 ==========
def format_axis(ax, ylabel, title, ymax):
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    # 标题字体放大，左对齐，距离图像留出呼吸空间
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    
    # Y轴留出额外 20% 的空间给数值标签
    ax.set_ylim(0, ymax * 1.25) 
    ax.tick_params(axis='y', labelsize=11)
    
    # 极简网格线：只保留Y轴虚线网格，并将层级放到底部 (zorder=0)
    ax.yaxis.grid(True, linestyle='--', color='#cccccc', alpha=0.7, zorder=0)
    ax.set_axisbelow(True) 
    
    # 移除顶部和右侧的边框 (Despine)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

format_axis(axes[0], 'Normalized Throughput', '(a) DNN Training Job Performance (Higher is better)', 0.6)
format_axis(axes[1], 'p99 Latency (s)', '(b) LLM Inference Job Performance (Lower is better)', 19.0)

# ========== 5. 图例的完美重构 ==========
handles, labels = axes[0].get_legend_handles_labels()

# 修改 LithOS 标签
for i in range(len(labels)):
    if labels[i] == 'LithOS':
        labels[i] = 'LithOS (No Semantic Determinism Guarantee)'

# 调整：将图例左对齐，0.06 的 X 坐标大致与子图的左侧边缘对齐
fig.legend(handles, labels, 
           loc='lower left', 
           bbox_to_anchor=(0.06, 0.96), 
           ncol=3, 
           frameon=False, 
           fontsize=12, 
           handlelength=2.5, 
           handleheight=1.2,
           columnspacing=2.0) # 增加列间距，避免文字拥挤

# 调整子图布局
plt.tight_layout()
# 微调 top 边距，确保左对齐的图例有足够空间
plt.subplots_adjust(top=0.9, hspace=0.45) 

# 导出为无损 PDF，bbox_inches='tight' 裁剪多余白边
plt.savefig('DNNTraining_with_LLMInference.pdf', bbox_inches='tight', format='pdf')
# plt.show()