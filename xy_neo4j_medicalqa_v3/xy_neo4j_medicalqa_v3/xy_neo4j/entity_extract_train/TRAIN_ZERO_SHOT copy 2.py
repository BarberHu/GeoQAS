import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# 使用更稳定的模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 初始化分类器
classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device="cpu"  # 使用CPU避免CUDA问题
)

def load_hydro_elements_from_csv(file_path):
    """
    从CSV文件加载水文要素。
    
    参数:
      file_path (str): CSV文件的路径。
    
    返回:
      List[str]: 从CSV文件加载的水文要素列表。
    """
    df = pd.read_csv(file_path)
    return df['Entity Name'].tolist()  # 确保列名匹配

def extract_hydrological_elements(user_question, labels, threshold=0.5):
    """
    利用零-shot分类从用户提问中提取相关的水文要素。
    
    参数:
      user_question (str): 用户的提问文本。
      labels (list): 候选的水文要素标签列表。
      threshold (float): 置信度阈值，只有得分超过该值的标签才认为是相关的。
    
    返回:
      List[Tuple[str, float]]: 返回提取到的标签及其对应的置信度。
    """
    try:
        result = classifier(
            user_question, 
            labels,
            hypothesis_template="这段文字包含{}这个概念。"  # 中文模板
        )
        relevant_elements = [
            (label, score) 
            for label, score in zip(result['labels'], result['scores']) 
            if score >= threshold
        ]
        return relevant_elements
    except Exception as e:
        print(f"提取过程出错: {str(e)}")
        return []

if __name__ == "__main__":
    print("程序开始运行...")
    try:
        # 加载数据
        hydro_elements = load_hydro_elements_from_csv(r"E:\毕业设计\参考\代码\Pycharm用\KG-LM Synergy项目实战\KG-LM Synergy项目实战\项目代码 - cursor\xy_neo4j_medicalqa_v3\xy_neo4j_medicalqa_v3\static\DATA_USE\zero_shot.csv")
        
        # 测试问题
        question = "我现在有黄河流域2005-2010年的降水数据和全国的DEM高程数据,现在我想对黄河流域进行径流模拟,请帮我设计一下建模步骤？"
        
        # 提取要素
        extracted = extract_hydrological_elements(question, hydro_elements)
        
        # 输出结果
        print("提取到的水文要素及其置信度：")
        for label, score in extracted:
            print(f"{label}: {score:.2f}")
            
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
