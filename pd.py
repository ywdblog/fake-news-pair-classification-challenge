# -*- coding: utf-8 -*
import pandas as pd
import sys 

str_gbk = "你好，世界".encode('gbk')

# 输出字节流
print(str_gbk)
print(type(str_gbk))

print(str_gbk.decode('utf-8'))


sys.exit(0)
s = b'\xe4\xb8\xad\xe6\x96\x87'
u = s.decode('utf-8')
print(u)  # 输出：中文

 
 
# 读取Excel文件
df = pd.read_csv('test.csv', nrows=100)

# 将前100行数据存储为CSV文件
df.to_csv('test-small.csv', index=False)