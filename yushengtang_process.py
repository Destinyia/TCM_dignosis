import pandas as pd
import json
import shutil
from pathlib import Path
import os

def process_yushengtang_data():
    """
    从Excel读取评估编号，复制图片并汇总JSON数据到CSV
    """
    # 定义路径
    excel_path = Path('datasets/yushengtang.xlsx')
    source_base_dir = Path('E:/Datasets/玉生堂四诊仪')
    output_base_dir = Path('datasets/yushengtang')
    csv_output = Path('datasets/yushengtang_dataTongue.csv')
    
    # 创建输出目录
    cutpic_dir = output_base_dir / 'CutPic'
    originpic_dir = output_base_dir / 'OriginPic'
    cutpic_dir.mkdir(parents=True, exist_ok=True)
    originpic_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取Excel文件获取评估编号
    df_excel = pd.read_excel(excel_path)
    # 假设评估编号在第一列，根据实际情况调整列名
    assessment_numbers = df_excel.iloc[:, 0].tolist()
    
    # 存储所有JSON数据
    all_data = []
    
    # 处理每个评估编号
    for assessment_num in assessment_numbers:
        print(f'Processing {assessment_num}...')
        
        # 构建源文件路径
        tongue_dir = source_base_dir / str(assessment_num) / 'tongue'
        
        if not tongue_dir.exists():
            print(f'  Warning: Directory not found - {tongue_dir}')
            continue
        
        # 复制CutPic
        cut_pic_source = tongue_dir / 'CutPic.jpg'
        if cut_pic_source.exists():
            cut_pic_dest = cutpic_dir / f'{assessment_num}.jpg'
            shutil.copy2(cut_pic_source, cut_pic_dest)
            print(f'  Copied CutPic')
        else:
            print(f'  Warning: CutPic.jpg not found')
        
        # 复制OriginPic
        origin_pic_source = tongue_dir / 'OriginPic.jpg'
        if origin_pic_source.exists():
            origin_pic_dest = originpic_dir / f'{assessment_num}.jpg'
            shutil.copy2(origin_pic_source, origin_pic_dest)
            print(f'  Copied OriginPic')
        else:
            print(f'  Warning: OriginPic.jpg not found')
        
        # 读取JSON数据
        json_file = tongue_dir / 'dataTongue.json'
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 添加评估编号作为索引
                data['AssessmentNumber'] = assessment_num
                all_data.append(data)
            print(f'  Read JSON data')
        else:
            print(f'  Warning: dataTongue.json not found')
    
    # 将所有数据转换为DataFrame并保存
    if all_data:
        df_result = pd.DataFrame(all_data)
        # 设置评估编号为索引
        df_result.set_index('AssessmentNumber', inplace=True)
        # 保存到CSV
        df_result.to_csv(csv_output, encoding='utf-8-sig', index=True)
        print(f'\nData saved to {csv_output}')
        print(f'Total records processed: {len(all_data)}')
    else:
        print('\nNo data to save')

if __name__ == '__main__':
    process_yushengtang_data()