# 地理建模智能问答系统

![系统标志](https://via.placeholder.com/150)

## 📑 目录

- [项目简介](#项目简介)
- [项目架构](#项目架构)
- [技术栈](#技术栈)
- [安装部署](#安装部署)
- [使用指南](#使用指南)
- [开发指南](#开发指南)
- [维护与更新](#维护与更新)
- [故障排除](#故障排除)
- [许可证信息](#许可证信息)

## 📖 项目简介

本项目是一个基于Neo4j知识图谱和多种大语言模型(LLM)的地理建模问答系统，专注于SWAT水文模型相关问题。系统能够接收用户问题，进行问题分解、知识图谱查询，并生成结构化回答。

### 主要特点

- **基于知识图谱**：利用Neo4j图数据库存储和查询专业领域知识
- **多LLM提供商支持**：集成DeepSeek、Zhipu和SiliconFlow等多种大语言模型
- **智能问题处理**：能够分解复杂问题，识别关键实体，进行渐进式回答
- **可视化功能**：支持知识图谱和处理流程的可视化展示
- **会话管理**：保存用户历史交互，支持导出和清除历史记录

### 应用场景

- 地理信息系统研究人员和学生的学习辅助工具
- SWAT水文模型相关问题的快速咨询系统
- 地理建模领域知识的智能检索和展示平台

## 🏗️ 项目架构

本系统采用Django Web框架作为后端，结合Neo4j图数据库和多种大语言模型API。整体架构如下：

```
                  +-------------+
                  |   Django    |
                  |   Views     |
                  +------+------+
                         |
                         v
           +---------------------------+
           |     DialogueManager       |
           +---------------------------+
           |                           |
           v                           v
 +-------------------+       +-------------------+
 | Entity Recognition |       |    LLM Client    |
 |    & Linking      |       |     Factory      |
 +-------------------+       +-------------------+
           |                           |
           v                           v
 +-------------------+       +-------------------+
 |   Neo4j Graph     |       |  Multiple LLM     |
 |   Database        |       |  API Providers    |
 +-------------------+       +-------------------+
```

### 核心组件说明

1. **DialogueManager**：对话管理器，系统的核心组件，负责协调各部分工作
2. **LLMClientFactory**：LLM客户端工厂，负责创建不同LLM提供商的客户端
3. **Entity_Mention模块**：负责实体识别和链接
4. **Neo4j数据库**：存储地理建模领域的知识图谱
5. **IntegratedQASystem**：集成问答系统，整合各组件功能

## 💻 技术栈

### 后端
- Python 3.7+
- Django 3.2.7
- Neo4j 图数据库

### 前端
- HTML/CSS/JavaScript
- Bootstrap（UI框架）
- D3.js（图形可视化）

### LLM集成
- DeepSeek API
- Zhipu API (智谱AI)
- SiliconFlow API

### 其他工具
- OpenAI兼容接口
- dotenv（环境变量管理）
- requests（HTTP请求）

## 🚀 安装部署

### 系统要求

- Python 3.7+
- Neo4j 数据库
- 足够的内存（推荐8GB+）
- 网络连接（用于API调用）

### 快速开始

1. **克隆代码库**：
   ```bash
   git clone <仓库地址>
   cd Geo_QAS/QAS
   ```

2. **创建虚拟环境**：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. **安装依赖**：
   ```bash
   pip install -r requests.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

4. **配置环境变量**：
   创建`.env`文件并配置以下环境变量：
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   DJANGO_SECRET_KEY=your-secret-key
   DJANGO_DEBUG=True
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ZHIPU_API_KEY=your_zhipu_api_key
   SILICONFLOW_API_KEY=your_siliconflow_api_key
   ```

5. **运行开发服务器**：
   ```bash
   python manage.py runserver
   ```
   
   系统在启动时会加载词表，这个过程会比较慢，需要几十秒，加载完成之后才能启动系统。

6. **访问系统**：
   在浏览器中打开 http://127.0.0.1:8000/

### 详细部署指南

#### 生产环境部署

请参考[部署文档](#维护与更新)部分，了解如何使用Gunicorn和Nginx部署生产环境。

## 📘 使用指南

### 系统功能概览

1. **问答功能**：
   - 输入与SWAT水文模型相关的问题
   - 系统自动分解问题，查询知识图谱，生成回答
   - 支持简单问题和复杂问题

2. **知识图谱可视化**：
   - 展示与问题相关的知识图谱
   - 支持图谱交互和探索

3. **会话管理**：
   - 查看历史会话记录
   - 导出会话内容
   - 清除历史记录

### 使用示例

**示例问题1**：SWAT模型中的水文循环组件有哪些？

**示例问题2**：SWAT模型如何模拟地表径流过程，需要哪些参数？

**示例问题3**：CN方法和SCS方法在SWAT模型中的区别是什么？各有什么优缺点？

### 界面说明

主界面包含以下部分：
- 问题输入区：输入您的问题
- 回答展示区：显示系统的回答
- 知识图谱区：展示相关知识的图谱可视化
- 历史记录区：查看历史会话

## 👨‍💻 开发指南

### 项目结构

```
Geo_QAS/
└── QAS/                       # 主项目目录
    ├── xy_neo4j/              # Django主应用
    ├── myneo4j/               # Neo4j数据库交互模块
    ├── templates/             # HTML模板
    ├── static/                # 静态文件(CSS/JS/图片)
    ├── accounts/              # 用户账户管理
    ├── llm_client_factory.py  # LLM客户端工厂
    ├── config.py              # 系统配置
    ├── config_loader.py       # 配置加载器
    ├── manage.py              # Django管理脚本
    └── README.md              # 项目说明文档
└── KG/                        # 知识图谱处理脚本
```

### 核心模块说明

1. **DialogueManager**：
   - 负责处理用户问题
   - 协调实体识别、知识图谱查询和LLM回答生成
   - 管理会话上下文

2. **LLMClientFactory**：
   - 创建不同LLM提供商的客户端
   - 管理API调用和错误处理
   - 支持OpenAI风格和自定义API调用

3. **Entity_Mention**：
   - 识别问题中的关键实体
   - 将实体链接到知识图谱中的节点

4. **Neo4j交互**：
   - 构建和执行Cypher查询
   - 处理查询结果
   - 知识图谱管理

### 添加新功能

#### 添加新的LLM提供商

1. 在`config.py`的`LLM_CONFIG["providers"]`中添加新提供商配置：
   ```python
   "new_provider": {
       "base_url": "https://api.new-provider.com",
       "default_model": "model-name",
       "max_tokens": 2048,
       "temperature": 0.7,
       "client_type": "openai"  # 或 "requests"
   }
   ```

2. 在`.env`文件中添加新提供商的API密钥。

3. 如果新提供商需要特殊处理，可能需要在`RequestsClient`或`OpenAIClient`类中添加特定逻辑，或创建新的客户端类。

#### 扩展问题处理流程

要添加新的问题处理步骤，修改`DialogueManager.get_response`方法，并添加相应的辅助方法。例如，添加情感分析步骤：

1. 创建情感分析方法：
   ```python
   def _analyze_sentiment(self, question):
       """分析问题的情感"""
       # 实现情感分析逻辑
       return sentiment_result
   ```

2. 在`get_response`方法中调用该方法。

### 代码风格指南

为保持代码一致性和可维护性，请遵循以下代码风格指南：

1. **命名规范**：
   - 类名：使用驼峰命名法（如`DialogueManager`）
   - 方法和变量：使用下划线命名法（如`get_response`）
   - 常量：全大写（如`API_KEYS`）

2. **文档字符串**：所有类和方法都应有文档字符串，描述功能、参数和返回值。

3. **错误处理**：使用try-except块处理可能的异常，并记录详细错误信息。

4. **代码组织**：
   - 相关函数放在一起
   - 辅助方法以下划线开头（如`_handle_api_error`）
   - 公共API不使用下划线前缀

## 🔧 维护与更新

### 部署步骤

#### 1. 准备服务器环境

**安装Python和依赖**：
```bash
# 更新系统
sudo apt update
sudo apt upgrade -y

# 安装Python和pip
sudo apt install python3 python3-pip python3-venv -y

# 安装必要工具
sudo apt install git nginx supervisor -y
```

**安装Neo4j数据库**：
```bash
# 添加Neo4j仓库
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update

# 安装Neo4j
sudo apt install neo4j -y

# 启动Neo4j服务
sudo systemctl start neo4j
sudo systemctl enable neo4j
```

#### 2. 部署应用

**克隆代码仓库**：
```bash
git clone <仓库地址> /opt/Geo_QAS
cd /opt/Geo_QAS/QAS
```

**设置虚拟环境并安装依赖**：
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requests.txt
```

**配置环境变量**：
```bash
# 创建.env文件
cat > .env << EOF
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
DJANGO_SECRET_KEY=your-secret-key
DJANGO_DEBUG=False
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
DEEPSEEK_API_KEY=your_deepseek_api_key
ZHIPU_API_KEY=your_zhipu_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
EOF
```

**设置数据库**：
```bash
python manage.py migrate
```

#### 3. 配置Gunicorn和Supervisor

**安装Gunicorn**：
```bash
pip install gunicorn
```

**创建Supervisor配置**：
```bash
sudo nano /etc/supervisor/conf.d/geo_qas.conf
```

添加以下内容：
```
[program:geo_qas]
command=/opt/Geo_QAS/QAS/venv/bin/gunicorn --workers 3 --bind unix:/opt/Geo_QAS/QAS/geo_qas.sock xy_neo4j.wsgi:application
directory=/opt/Geo_QAS/QAS
user=www-data
group=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/geo_qas/error.log
stdout_logfile=/var/log/geo_qas/access.log
```

**创建日志目录**：
```bash
sudo mkdir -p /var/log/geo_qas
sudo chown -R www-data:www-data /var/log/geo_qas
```

**启动服务**：
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status
```

#### 4. 配置Nginx

**创建Nginx配置**：
```bash
sudo nano /etc/nginx/sites-available/geo_qas
```

添加以下内容：
```
server {
    listen 80;
    server_name your_domain.com www.your_domain.com;

    location = /favicon.ico { access_log off; log_not_found off; }
    location /static/ {
        root /opt/Geo_QAS/QAS;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/opt/Geo_QAS/QAS/geo_qas.sock;
    }
}
```

**启用站点并重启Nginx**：
```bash
sudo ln -s /etc/nginx/sites-available/geo_qas /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

#### 5. 设置SSL（可选但推荐）

使用Let's Encrypt设置SSL：
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your_domain.com -d www.your_domain.com
```

### 日常维护

#### 定期更新代码

```bash
cd /opt/Geo_QAS/QAS
git pull
source venv/bin/activate
pip install -r requests.txt
python manage.py migrate
sudo supervisorctl restart geo_qas
```

#### 日志监控

查看应用日志：
```bash
sudo tail -f /var/log/geo_qas/error.log
sudo tail -f /var/log/geo_qas/access.log
```

查看Nginx日志：
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

#### 数据备份

备份Neo4j数据库：
```bash
sudo systemctl stop neo4j
sudo tar -czf neo4j_backup_$(date +%Y%m%d).tar.gz /var/lib/neo4j/data
sudo systemctl start neo4j
```

备份Django SQLite数据库：
```bash
cp /opt/Geo_QAS/QAS/db.sqlite3 ~/db_backup_$(date +%Y%m%d).sqlite3
```

## ❓ 故障排除

### 常见问题及解决方案

1. **系统无法启动**：
   - 检查环境变量是否正确设置
   - 检查Django错误日志
   - 确保所有依赖已正确安装

2. **API调用失败**：
   - 验证API密钥是否有效
   - 检查网络连接
   - 查看错误日志以获取具体错误信息

3. **Neo4j连接问题**：
   - 确保Neo4j服务正在运行
   - 验证Neo4j凭据是否正确
   - 检查Neo4j日志中的错误

4. **性能问题**：
   - 增加Gunicorn工作进程数量
   - 优化数据库查询
   - 考虑增加服务器资源

### 错误代码说明

| 错误代码 | 描述 | 解决方法 |
|---------|------|---------|
| E001 | API密钥无效 | 检查并更新.env文件中的API密钥 |
| E002 | Neo4j连接失败 | 确保Neo4j服务运行并检查连接参数 |
| E003 | LLM调用超时 | 检查网络连接，可能需要增加超时时间 |
| E004 | 实体识别失败 | 检查相关模型是否正确加载 |

## 📄 许可证信息

本项目采用MIT许可证。详细信息请查看[LICENSE](LICENSE)文件。

### 第三方许可

本项目使用了以下第三方库和工具，感谢他们的贡献：

- Django - BSD许可证
- Neo4j - GNU通用公共许可证
- Bootstrap - MIT许可证
- D3.js - BSD许可证

---

© 2024 地理建模智能问答系统团队。保留所有权利。