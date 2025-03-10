# 从config.py导入所有配置项
from config import *

# 如果需要在这里可以进行任何其他初始化 

# 替换
from xy_neo4j_medicalqa_v3.config import API_KEYS, LLM_CONFIG, ENTITY_CONFIG

# 为
from config_loader import API_KEYS, LLM_CONFIG, ENTITY_CONFIG 