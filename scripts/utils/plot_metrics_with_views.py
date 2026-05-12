import matplotlib.pyplot as plt
import numpy as np

# Data from the table
views = [200, 160, 120, 80, 40, 20, 0]
metrics_data_incre_viewcov = {
    'mPre @25%': [34.5, 32.4, 30.3, 27.2, 19.7, 17.1, 0.],
    # 'mPre @50%': [21.2, 18.6, 17.7, 16.7, 13.0, 10.4, 0.],
    # 'mPre All': [8.5, 7.3, 7.2, 6.0, 4.5, 4.3, 0.],
    # 'mIoU': [26.5, 23.7, 22.1, 19.9, 14.7, 12.4, 0.],
    'mAcc': [32.2, 31.0, 29.5, 25.6, 19.7, 15.9, 0.], 
    'Queries / Instance': [8.6, 8.1, 7.5, 7.0, 6.0, 4.8, 0]
}
metrics_data_incre_vis = {
    'mPre @25%': [36.4, 33.6, 29.6, 25.1, 19.0, 17.4, 0.],
    # 'mPre @50%': [22.0, 18.6, 15.5, 13.4, 11.3, 10.2, 0.],
    # 'mPre All': [8.6, 7.2, 6.5, 5.4, 4.3, 4.2, 0.],
    # 'mIoU': [26.9, 25.0, 22.8, 19.7, 14.3, 11.7, 0.],
    'mAcc': [33.2, 31.6, 29.0, 24.7, 19.2, 15.6, 0.], 
    'Queries / Instance': [18.7, 17.3, 15.6, 14.2, 11.7, 9.4, 0]
}
metric_data_overall = {
    'mPre @25%': [36.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
    # 'mPre @50%': [22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
    # 'mPre All': [8.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
    # 'mIoU': [26.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
    'mAcc': [33.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.]
}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Incremental Performance Metrics', fontsize=16, fontweight='bold')

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Colors for each metric
c = ['#2E86C1', '#F39C12', '#28B463', '#8E44AD',  '#E74C3C', "#39D1D4"]

# Plot each metric
for i, (metric_name, values) in enumerate(metrics_data_incre_viewcov.items()):
    ax = axes_flat[i]
    
    # Create line plot with markers
    
    ax.plot(views, metrics_data_incre_vis[metric_name], 
        marker='x', markersize=8, linestyle='-', linewidth=2, 
        color=c[i], markerfacecolor='white', 
        markeredgecolor=c[i], markeredgewidth=2, 
        label='Incre. Top-8 Visibility'
    )
    ax.plot(views, values, 
        marker='o', markersize=8, linestyle='-', linewidth=2, 
        color=c[i], markerfacecolor='white', 
        markeredgecolor=c[i], markeredgewidth=2, 
        label='Incre. View Coverage'
    )
    if metric_name != 'Queries / Instance':
        ax.plot([200] + views, metric_data_overall[metric_name], 
            marker='x', markersize=8, linestyle='--', linewidth=1, 
            color=c[i], markerfacecolor='white', 
            markeredgecolor=c[i], markeredgewidth=2, 
            label='Post Top-8 Visibility'
        )

    # Customize the subplot
    # ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Observed Frames', fontsize=12)
    if metric_name == 'Queries / Instance':
        ax.set_ylabel(f'{metric_name}', fontsize=12, fontweight='bold', c='red')
    ax.set_ylabel(f'{metric_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(30, 210)
    
    # # Add value labels on points
    # for x, y in zip(views, values):
    #     ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", 
    #                xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    # Set x-axis ticks
    ax.set_xticks(views)
    ax.set_xticklabels(views)
    
    # Add background color
    ax.set_facecolor('#f8f9fa')
    ax.legend(loc='upper left', fontsize=10)


# Adjust layout
plt.tight_layout()
plt.show()