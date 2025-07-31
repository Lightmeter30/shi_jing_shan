import os
import sys
from loguru import logger
from datetime import datetime

# 确保日志目录存在
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 移除默认的处理器
logger.remove()

# 添加控制台输出
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    serialize=False
)

# 添加文件输出
logger.add(
    os.path.join(log_dir, "project.log"),
    rotation="500 MB",
    retention="7 days",
    encoding="utf-8",
    enqueue=True,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    serialize=False
)

# 添加错误日志文件
logger.add(
    os.path.join(log_dir, "error.log"),
    rotation="100 MB",
    retention="30 days",
    encoding="utf-8",
    enqueue=True,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    serialize=False
)