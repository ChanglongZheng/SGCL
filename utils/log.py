import os
import sys
import re
# \x1b[：Control Sequence Introducer  后面是一个正则表达式
# 感觉这个是因为要在linux终端输出带颜色字体，所以这样设置，后边的sub也是为了还原信息
pattern = re.compile(r'\x1b\[[0-9;]*m')


class Logger(object):
    def __init__(self, path):
        log_file = os.path.join(path, 'training.log')
        print('saving log to : ', path)
        self.terminal = sys.stdout  # sys.stdout.write()直接往terminal输出，和print()类似
        self.file = None
        self.open(log_file)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w+'  # a则是追加 w+是覆盖
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        """ 向终端或者文件中写入数据 """
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(pattern.sub('',message))  # substitute匹配到的字符串为空字符串
            self.file.flush()  # flush，确保数据被写入

    def close(self):
        self.file.close()

