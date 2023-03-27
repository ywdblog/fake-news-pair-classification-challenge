# -*- coding: utf-8 -*

import pandas as pd

# 读取Excel文件
df = pd.read_csv('test.csv', nrows=100)

# 将前100行数据存储为CSV文件
df.to_csv('test-small.csv', index=False)