# -*- coding: utf-8 -*-
# @Time : 2023/5/23 20:03
# @Author : DanYang
# @File : main.py
# @Software : PyCharm
import time

from crawler.Crawler import Crawler

PAGE = 89


if __name__ == '__main__':
    c = Crawler()
    for i in range(1, PAGE+1):
        c.scheduling(i)
        time.sleep(5)
