import os
import json
import time
from typing import Dict

# 配置信息
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "wswy0129"
}

LLM_API_KEY = "sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"

def main():
    """主函数 - 问答系统演示"""
    # 初始化系统
    from integrated_qa_system import IntegratedQASystem
    
    print("正在初始化问答系统...")
    start_time = time.time()
    qa_system = IntegratedQASystem(
        neo4j_config=NEO4J_CONFIG,
        llm_api_key=LLM_API_KEY
    )
    init_time = time.time() - start_time
    print(f"系统初始化完成，耗时: {init_time:.2f}秒")
    
    # 测试简单问题
    simple_question = "SWAT模型在径流模拟中如何利用气象数据?"
    print(f"\n问题: {simple_question}")
    
    start_time = time.time()
    answer = qa_system.answer_question(simple_question, use_decomposition=False)
    query_time = time.time() - start_time
    
    print(f"回答: {answer}")
    print(f"回答时间: {query_time:.2f}秒")
    
    # 测试复杂问题
    complex_question = "我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河流域进行径流模拟?"
    print(f"\n复杂问题: {complex_question}")
    
    start_time = time.time()
    answer = qa_system.answer_question(complex_question, use_decomposition=True)
    query_time = time.time() - start_time
    
    print(f"回答: {answer}")
    print(f"回答时间: {query_time:.2f}秒")
    
    # 保存结果到文件
    results = {
        "simple_question": {
            "question": simple_question,
            "answer": answer,
            "time": query_time
        },
        "complex_question": {
            "question": complex_question,
            "answer": answer,
            "time": query_time
        }
    }
    
    with open("qa_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n测试完成，结果已保存到 qa_results.json")

if __name__ == "__main__":
    main()