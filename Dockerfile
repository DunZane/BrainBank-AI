# 使用官方的Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app/brainbank

# 将环境变量添加到容器中
ENV OPENAI_API_KEY="sk-zcFbFTpmRMwEjSQaD0152b150c114f7c8cE8AbE20e31C355"
ENV OPENAI_BASE_URL="https://api.xiaoai.plus/v1"

# 复制当前目录内容到容器中的/app目录
COPY . /app/brainbank

# 安装所需的Python包
RUN pip install --no-cache-dir -r requirements.txt

# 暴露应用程序运行的端口
EXPOSE 8111

# 运行Python应用程序
CMD ["python", "/app/brainbank/app.py"]