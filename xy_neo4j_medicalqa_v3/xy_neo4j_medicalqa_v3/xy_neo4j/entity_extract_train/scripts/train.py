// scripts/train.py
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
import json
import os

# 加载数据集
def load_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 定义数据预处理函数
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["input"],
        truncation=True,
        max_length=256,
        is_split_into_words=False,
        padding='max_length'
    )
    
    # 这里需要根据您的标注方案调整
    labels = []
    for entity in examples["output"]["核心实体"]:
        # 示例标注逻辑，需要根据实际需求调整
        label = [0] * len(tokenized_inputs["input_ids"])
        # 假设实体出现在问题开头
        label[1] = 1  # B-ENT
        label[2] = 2  # I-ENT
        labels.append(label)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def train():
    # 加载预训练模型
    model = BertForTokenClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=3  # 根据实体类型数量调整
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    
    # 加载训练数据
    dataset = load_dataset(os.path.join(os.path.dirname(__file__), '../data/train.json'))
    
    # 数据预处理
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir='./output',
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        fp16=True
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

if __name__ == "__main__":
    train()
