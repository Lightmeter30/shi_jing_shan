import time
import numpy as np
import uuid
import os

def getNewName(file_type):
    # 前面是file_type+年月日时分秒
    new_name = time.strftime(file_type+'-%Y%m%d%H%M%S', time.localtime())
    # 最后是5个随机数字
    # Python中的numpy库中的random.randint(a, b, n)表示随机生成n个大于等于a，小于b的整数
    ranlist = np.random.randint(0, 10, 5)
    for i in ranlist:
        new_name += str(i)
    # 加后缀名
    new_name += '.jpg'
    # 返回字符串
    return new_name

def new_name(dir, filename):
  ext = filename.split('.')[-1]
  filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
  # return the whole path to the file
  return os.path.join(dir, filename)