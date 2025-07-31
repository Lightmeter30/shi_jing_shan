import time
import functools
from typing import Callable, Any
from .logger_config import logger

def timer(func: Callable) -> Callable:
    """
    装饰器：记录函数运行时间
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"函数 {func.__name__} 运行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper
