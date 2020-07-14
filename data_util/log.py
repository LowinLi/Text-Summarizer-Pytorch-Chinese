"""
日志模板相关
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建 handler 输出到文件
root = os.path.dirname(__file__)
if 'logs' not in os.listdir(root):
    os.mkdir(root + '/logs/')
    logger.info('日志目录:' + root + '/logs/')
name = root + "/logs/%s.log" % datetime.now().strftime('%Y%m%d')

handler = logging.FileHandler(name)
handler.setLevel(logging.INFO)

# handler 输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# 创建 logging format
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(ch)

logger.info('log启动')
