import json
import os
import glob

def process_json_file(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    def remove_source_article(obj):
        if isinstance(obj, dict):
            # 如果不是地理问题类型，删除source_article字段
            if obj.get('type') != '地理问题' and 'source_article' in obj:
                del obj['source_article']
            
            # 递归处理所有值
            for value in obj.values():
                remove_source_article(value)
        elif isinstance(obj, list):
            # 递归处理列表中的所有项
            for item in obj:
                remove_source_article(item)
    
    # 处理JSON数据
    remove_source_article(data)
    
    # 保存修改后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # 获取目录中的所有JSON文件
    json_files = glob.glob('xy_neo4j_medicalqa_v3/KG/output0320/*.json')
    
    # 处理每个文件
    for file_path in json_files:
        if file_path.endswith('.json'):
            print(f'处理文件: {file_path}')
            try:
                process_json_file(file_path)
                print(f'成功处理: {file_path}')
            except Exception as e:
                print(f'处理文件 {file_path} 时出错: {str(e)}')

if __name__ == '__main__':
    main() 