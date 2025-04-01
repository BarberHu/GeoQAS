# 地理建模智能问答系统

## 环境安装
- python=3.7+
- Django = 3.2.7

## 安装方法
- pip install -r requests.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
- python manage.py runserver 
- 浏览器打开 http://127.0.0.1:8000/

系统在启动时会加载词表，这个过程会比较慢，需要几十秒，加载完成之后才能启动系统。

## 系统简介

本项目是一个基于Neo4j知识图谱和多种大语言模型(LLM)的地理建模问答系统，专注于SWAT水文模型相关问题。系统能够接收用户问题，进行问题分解、知识图谱查询，并生成结构化回答。

### 主要功能

1. **多LLM提供商支持**：支持DeepSeek、Zhipu和SiliconFlow等多种LLM
2. **问题处理流程**：
   - 问题分解：将复杂问题分解为子问题
   - 实体识别：识别问题中的关键实体
   - 知识图谱查询：根据实体查询Neo4j图数据库
   - 渐进式回答生成：先回答各子问题，再综合生成最终答案
3. **可视化功能**：支持知识图谱和流程图可视化
4. **会话管理**：保存用户历史交互，支持导出和清除历史记录

## 开发者指南

### 系统架构

本系统采用Django Web框架，结合Neo4j图数据库和多种大语言模型。主要组件包括：

1. **DialogueManager**：对话管理器，系统的核心组件，负责协调各部分工作
2. **LLMClientFactory**：LLM客户端工厂，负责创建不同LLM提供商的客户端
3. **Entity_Mention模块**：负责实体识别和链接
4. **IntegratedQASystem**：集成问答系统

系统架构图：
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

### 开发环境设置

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
   创建.env文件并配置以下环境变量：
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ZHIPU_API_KEY=your_zhipu_api_key
   SILICONFLOW_API_KEY=your_siliconflow_api_key
   ```

5. **运行开发服务器**：
   ```bash
   python manage.py runserver
   ```

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

5. **减少重复代码**：
   - 将重复逻辑抽取到公共函数或基类中
   - 使用继承和组合减少代码重复

6. **日志记录**：使用print语句记录关键操作和错误，便于调试。

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

2. 在`API_KEYS`中添加新提供商的API密钥。

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

## 部署文档

### 系统要求

- Python 3.7+
- Django 3.2.7
- Neo4j 数据库
- 足够的内存（推荐8GB+）和处理能力
- 网络连接（用于API调用）

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
sudo nano /etc/supervisor/conf.d/xy_neo4j_qa.conf
```

添加以下内容：
```
[program:xy_neo4j_qa]
command=/opt/Geo_QAS/QAS/venv/bin/gunicorn --workers 3 --bind unix:/opt/Geo_QAS/QAS/xy_neo4j_qa.sock xy_neo4j.wsgi:application
directory=/opt/Geo_QAS/QAS
user=www-data
group=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/xy_neo4j_qa/error.log
stdout_logfile=/var/log/xy_neo4j_qa/access.log
```

**创建日志目录**：
```bash
sudo mkdir -p /var/log/xy_neo4j_qa
sudo chown -R www-data:www-data /var/log/xy_neo4j_qa
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
sudo nano /etc/nginx/sites-available/xy_neo4j_qa
```

添加以下内容：
```
server {
    listen 80;
    server_name your_domain.com;

    location = /favicon.ico { access_log off; log_not_found off; }
    location /static/ {
        root /opt/Geo_QAS/QAS;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/opt/Geo_QAS/QAS/xy_neo4j_qa.sock;
    }
}
```

**启用站点并重启Nginx**：
```bash
sudo ln -s /etc/nginx/sites-available/xy_neo4j_qa /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

#### 5. 设置SSL（可选但推荐）

使用Let's Encrypt设置SSL：
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your_domain.com
```

### 维护与更新

#### 定期更新代码

```bash
cd /opt/Geo_QAS/QAS
git pull
source venv/bin/activate
pip install -r requests.txt
python manage.py migrate
sudo supervisorctl restart xy_neo4j_qa
```

#### 日志监控

查看应用日志：
```bash
sudo tail -f /var/log/xy_neo4j_qa/error.log
sudo tail -f /var/log/xy_neo4j_qa/access.log
```

查看Nginx日志：
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

#### 备份数据

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

## 故障排除

### 常见问题

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

如有其他问题，请联系开发团队或查阅系统日志以获取更多信息。





