import warnings
# 忽略特定的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

from integrated_qa_system import IntegratedQASystem

def main():
    # Neo4j配置
    NEO4J_CONFIG = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "wswy0129"
    }
    
    # 初始化系统
    qa_system = IntegratedQASystem(
        neo4j_config=NEO4J_CONFIG,
        llm_api_key="sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"
    )
    
    # 单轮问答示例
    question = "SWAT模型在径流模拟中如何利用气象数据?"
    print(f"问题: {question}")
    answer = qa_system.answer_question(question, use_decomposition=False)
    print(f"回答: {answer}")
    
    # 复杂问题示例
    complex_question = "我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河流域进行径流模拟?"
    print(f"\n复杂问题: {complex_question}")
    answer = qa_system.answer_question(complex_question, use_decomposition=True)
    print(f"回答: {answer}")

if __name__ == "__main__":
    main()