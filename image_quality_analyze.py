import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_brightness(image):
    """计算图片平均亮度"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return np.mean(gray)


def calculate_contrast(image):
    """计算图片对比度（标准差）"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return np.std(gray)


def calculate_sharpness(image):
    """计算图片清晰度（使用Laplacian方差）"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def calculate_saturation(image):
    """计算舌色饱和度（HSV空间中S通道的平均值）"""
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 1])
    return 0.0


def get_time_period(time_str):
    """根据时间划分早中晚"""
    try:
        dt = pd.to_datetime(time_str)
        hour = dt.hour
        if 6 <= hour < 12:
            return '早 (06:00-12:00)'
        elif 12 <= hour < 18:
            return '中 (12:00-18:00)'
        else:
            return '晚 (18:00-06:00)'
    except:
        return None


def analyze_image_quality(image_path):
    """分析单张图片的质量指标"""
    if not image_path.exists():
        return None
    
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    return {
        'brightness': calculate_brightness(image),
        'contrast': calculate_contrast(image),
        'sharpness': calculate_sharpness(image),
        'saturation': calculate_saturation(image),
    }


def analyze_dataset_quality(limit=None):
    """分析整个数据集的图片质量（仅分析CutPic）"""
    print('Loading data...')
    df = pd.read_csv('datasets/yushengtang.csv', dtype={'AssessmentNumber': str})
    df = df.drop_duplicates(subset=['AssessmentNumber'])
    
    if limit:
        df = df.head(limit)
        print(f'Testing mode: Analyzing only first {limit} samples\n')
    
    output_dir = Path('runs/quality_analyze')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f'Analyzing {len(df)} samples (CutPic only)...')
    for idx, row in df.iterrows():
        assessment_num = str(row['AssessmentNumber'])
        
        if len(results) % 100 == 0 and len(results) > 0:
            print(f'  Progress: {len(results)}/{len(df)}')
        
        customer_name = row.get('客户名称', 'Unknown')
        assessment_time = row.get('评估时间', 'Unknown')
        time_period = get_time_period(assessment_time)
        
        cut_pic_path = Path(f'datasets/yushengtang/CutPic/{assessment_num}.jpg')
        cut_quality = analyze_image_quality(cut_pic_path)
        
        if cut_quality:
            results.append({
                '评估编号': assessment_num,
                '客户名称': customer_name,
                '评估时间': assessment_time,
                '时间段': time_period,
                '亮度': cut_quality['brightness'],
                '对比度': cut_quality['contrast'],
                '清晰度': cut_quality['sharpness'],
                '舌色饱和度': cut_quality['saturation'],
            })
    
    df_results = pd.DataFrame(results)
    
    detail_csv = output_dir / 'quality_analysis_detailed.csv'
    df_results.to_csv(detail_csv, index=False, encoding='utf-8-sig')
    print(f'\nDetailed results saved to: {detail_csv}')
    
    print('\n=== Quality Statistics ===')
    print(df_results[['亮度', '对比度', '清晰度', '舌色饱和度']].describe())
    
    visualize_quality_analysis(df_results, output_dir)
    analyze_by_customer(df_results, output_dir)
    analyze_by_time(df_results, output_dir)
    
    return df_results


def visualize_quality_analysis(df, output_dir):
    """可视化质量分析结果"""
    print('\nGenerating visualizations...')
    
    metrics = ['亮度', '对比度', '清晰度', '舌色饱和度']
    
    # 1. 整体质量指标分布（叠加最高、最低客户分布）
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # 获取该指标最高和最低的客户
        customer_metric = df.groupby('客户名称')[metric].mean().sort_values()
        best_customer = customer_metric.index[-1]
        worst_customer = customer_metric.index[0]
        
        # 总体分布
        data_all = df[metric]
        # 对清晰度使用对数变换
        if metric == '清晰度':
            # 过滤掉0或负值
            data_all_plot = np.log10(data_all[data_all > 0] + 1)
            data_best = np.log10(df[df['客户名称'] == best_customer][metric] + 1)
            data_worst = np.log10(df[df['客户名称'] == worst_customer][metric] + 1)
            xlabel = f'{metric} (log10)'
        else:
            data_all_plot = data_all
            data_best = df[df['客户名称'] == best_customer][metric]
            data_worst = df[df['客户名称'] == worst_customer][metric]
            xlabel = metric
        
        # 绘制直方图
        ax.hist(data_all_plot, alpha=0.5, bins=40, edgecolor='black', 
               label=f'总体 (n={len(data_all)})', color='gray')
        ax.hist(data_best, alpha=0.7, bins=20, edgecolor='darkgreen', 
               label=f'最高: {best_customer} (n={len(data_best)})', color='green')
        ax.hist(data_worst, alpha=0.7, bins=20, edgecolor='darkred', 
               label=f'最低: {worst_customer} (n={len(data_worst)})', color='red')
        
        # 添加均值线
        ax.axvline(data_all_plot.mean(), color='blue', linestyle='--', 
                  label=f'总体平均', linewidth=2)
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title(f'{metric}分布对比 (总体 vs 最高/最低客户)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: distribution_overall.png')
    
    # 2. 按时间段的质量指标箱线图
    df_time = df[df['时间段'].notna()]
    if len(df_time) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        time_order = ['早 (06:00-12:00)', '中 (12:00-18:00)', '晚 (18:00-06:00)']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            df_time_sorted = df_time[df_time['时间段'].isin(time_order)]
            
            sns.boxplot(data=df_time_sorted, x='时间段', y=metric, 
                       ax=ax, order=time_order)
            ax.set_xlabel('时间段')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric}按时间段对比')
            ax.tick_params(axis='x', rotation=15)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_by_timeperiod.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: quality_by_timeperiod.png')
    
    # 3. 质量指标相关性热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr = df[metrics].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, ax=ax, vmin=-1, vmax=1, square=True)
    ax.set_title('质量指标相关性热力图')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: correlation_heatmap.png')


def analyze_by_customer(df, output_dir):
    """按客户名称分析"""
    print('\nAnalyzing by customer...')
    
    metrics = ['亮度', '对比度', '清晰度', '舌色饱和度']
    
    customer_stats = df.groupby('客户名称')[metrics].agg(['mean', 'std', 'count'])
    customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns.values]
    customer_stats = customer_stats.sort_values('清晰度_mean', ascending=False)
    customer_stats.to_csv(output_dir / 'quality_by_customer.csv', encoding='utf-8-sig')
    print(f'  Saved: quality_by_customer.csv')
    
    # 找出质量最好和最差的客户
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        customer_metric = df.groupby('客户名称')[metric].mean().sort_values()
        
        n = min(10, len(customer_metric) // 2)
        combined = pd.concat([customer_metric.head(n), customer_metric.tail(n)])
        
        colors = ['red'] * n + ['green'] * n
        ax.barh(range(len(combined)), combined.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(combined.index, fontsize=9)
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} - 最低/最高客户对比')
        ax.axvline(df[metric].mean(), color='blue', linestyle='--', 
                  label=f'平均值: {df[metric].mean():.2f}', linewidth=2)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_by_customer_top_bottom.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: quality_by_customer_top_bottom.png')
    
    # 保存Top/Bottom数据到Excel
    top_bottom_data = {}
    for metric in metrics:
        customer_metric = df.groupby('客户名称')[metric].agg(['mean', 'count']).sort_values('mean')
        top_bottom_data[f'{metric}_Bottom5'] = customer_metric.head(5)
        top_bottom_data[f'{metric}_Top5'] = customer_metric.tail(5)
    
    with pd.ExcelWriter(output_dir / 'quality_top_bottom.xlsx', engine='openpyxl') as writer:
        for key, data in top_bottom_data.items():
            data.to_excel(writer, sheet_name=key[:31])
    print(f'  Saved: quality_top_bottom.xlsx')


def analyze_by_time(df, output_dir):
    """按评估时间分析（早中晚时间段）"""
    print('\nAnalyzing by time period...')
    
    df_time = df[df['时间段'].notna()]
    
    if len(df_time) == 0:
        print('  No valid time data found.')
        return
    
    time_order = ['早 (06:00-12:00)', '中 (12:00-18:00)', '晚 (18:00-06:00)']
    metrics = ['亮度', '对比度', '清晰度', '舌色饱和度']
    
    time_stats = df_time[df_time['时间段'].isin(time_order)].groupby('时间段')[metrics].agg(['mean', 'std', 'count'])
    time_stats.columns = ['_'.join(col).strip() for col in time_stats.columns.values]
    time_stats = time_stats.reindex(time_order)
    time_stats.to_csv(output_dir / 'quality_by_time.csv', encoding='utf-8-sig')
    print(f'  Saved: quality_by_time.csv')
    
    print('\n样本数按时间段分布:')
    for period in time_order:
        if period in time_stats.index:
            count = time_stats.loc[period, '亮度_count']
            print(f'  {period}: {int(count)} 个样本')
    
    # 按月份统计
    df_time_copy = df.copy()
    df_time_copy['评估时间'] = pd.to_datetime(df_time_copy['评估时间'], errors='coerce')
    df_time_copy = df_time_copy.dropna(subset=['评估时间'])
    
    if len(df_time_copy) > 0:
        df_time_copy['月份'] = df_time_copy['评估时间'].dt.to_period('M')
        monthly_stats = df_time_copy.groupby('月份')[metrics].agg(['mean', 'std', 'count'])
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
        monthly_stats.to_csv(output_dir / 'quality_by_month.csv', encoding='utf-8-sig')
        print(f'  Saved: quality_by_month.csv')


def generate_quality_report(df, output_dir):
    """生成质量分析报告"""
    print('\nGenerating quality report...')
    
    report = []
    report.append('=' * 80)
    report.append('图片质量分析报告 (CutPic)')
    report.append('=' * 80)
    report.append(f'分析时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report.append(f'总样本数: {len(df)}')
    report.append(f'客户数量: {df["客户名称"].nunique()}')
    report.append('')
    
    # 整体统计
    report.append('1. 整体质量统计')
    report.append('-' * 80)
    metrics = ['亮度', '对比度', '清晰度', '舌色饱和度']
    for metric in metrics:
        report.append(f'\n{metric}:')
        report.append(f'  平均值: {df[metric].mean():.2f}')
        report.append(f'  标准差: {df[metric].std():.2f}')
        report.append(f'  最小值: {df[metric].min():.2f}')
        report.append(f'  最大值: {df[metric].max():.2f}')
    
    low_sharpness = df[df['清晰度'] < df['清晰度'].quantile(0.1)]
    report.append(f'\n低清晰度图片数 (< 10%分位数): {len(low_sharpness)}')
    
    # 按时间段统计
    report.append('')
    report.append('2. 按时间段统计')
    report.append('-' * 80)
    df_time = df[df['时间段'].notna()]
    if len(df_time) > 0:
        time_order = ['早 (06:00-12:00)', '中 (12:00-18:00)', '晚 (18:00-06:00)']
        for period in time_order:
            df_period = df_time[df_time['时间段'] == period]
            if len(df_period) > 0:
                report.append(f'\n{period}:')
                report.append(f'  样本数: {len(df_period)}')
                report.append(f'  平均亮度: {df_period["亮度"].mean():.2f}')
                report.append(f'  平均清晰度: {df_period["清晰度"].mean():.2f}')
                report.append(f'  平均舌色饱和度: {df_period["舌色饱和度"].mean():.2f}')
    
    # Top和Bottom客户
    report.append('')
    report.append('3. 客户质量分析 (按清晰度)')
    report.append('-' * 80)
    customer_sharpness = df.groupby('客户名称')['清晰度'].agg(['mean', 'count']).sort_values('mean')
    
    report.append('\n清晰度最低的5个客户:')
    for idx, (customer, row) in enumerate(customer_sharpness.head(5).iterrows(), 1):
        report.append(f'  {idx}. {customer}: {row["mean"]:.2f} ({int(row["count"])} 样本)')
    
    report.append('\n清晰度最高的5个客户:')
    for idx, (customer, row) in enumerate(customer_sharpness.tail(5).iterrows(), 1):
        report.append(f'  {idx}. {customer}: {row["mean"]:.2f} ({int(row["count"])} 样本)')
    
    report.append('')
    report.append('4. 数据质量建议')
    report.append('-' * 80)
    
    avg_sharpness = df['清晰度'].mean()
    if avg_sharpness < 100:
        report.append(f'  ⚠️  平均清晰度较低 ({avg_sharpness:.2f})，建议检查图片采集设备或拍摄条件')
    else:
        report.append(f'  ✓  平均清晰度良好 ({avg_sharpness:.2f})')
    
    avg_saturation = df['舌色饱和度'].mean()
    if avg_saturation < 50:
        report.append(f'  ⚠️  平均舌色饱和度较低 ({avg_saturation:.2f})，建议检查光照条件')
    else:
        report.append(f'  ✓  平均舌色饱和度正常 ({avg_saturation:.2f})')
    
    report.append('')
    report.append('=' * 80)
    report.append('报告结束')
    report.append('=' * 80)
    
    report_path = output_dir / 'quality_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f'  Saved: quality_report.txt')
    print('\n' + '\n'.join(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze tongue image quality (CutPic only)')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of samples for testing')
    args = parser.parse_args()
    
    print('Starting image quality analysis (CutPic only)...\n')
    df_results = analyze_dataset_quality(limit=args.limit)
    generate_quality_report(df_results, Path('runs/quality_analyze'))
    print('\n✓ Analysis complete!')
