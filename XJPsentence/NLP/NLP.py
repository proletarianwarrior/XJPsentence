# -*- coding: utf-8 -*-
# @Time : 2023/5/26 1:03
# @Author : DanYang
# @File : NLP.py
# @Software : PyCharm
import os
import re
import tiktoken
import time

import numpy as np
from retrying import retry
import requests.exceptions as ep
import openai
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

client = MongoClient("mongodb://localhost:27017/")
db = client['XJPdata']
collection = db['diplomats']


def _load_data(condition=None, limit=886):
    if condition:
        return list(collection.find(condition).limit(limit))
    else:
        return list(collection.find().limit(limit))


def _keep_chinese(text):
    pattern = re.compile(r'[^\u4e00-\u9fa5，。！？；：“”‘’【】（）《》\n]+')
    chinese_text = re.sub(pattern, '', text)
    chinese_text = re.sub(r'\s', '', chinese_text)
    return chinese_text


def _cal_token(message, model, prompt):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt.format_prompt(describe=message).to_string()))


class GPT:
    def __init__(self, temperature=0, model='gpt-3.5-turbo'):
        load_dotenv('OPENAI_KEY.env')
        openai.proxy = 'http://127.0.0.1:4780'
        api_key = os.getenv('API_KEY')
        self.model = model
        self.llm = ChatOpenAI(
            temperature=temperature,
            openai_api_key=api_key,
            model_name=model
        )
        self.prompt = PromptTemplate(
            input_variables=["describe"],
            template="""
            这是一篇有关习近平外交的新闻报道：{describe}。
            请为我总结最多10个关键词(具体几个合适你自己判断)并给出关键词在文章中的权重值(范围为0到1)，输出格式为：关键词+英文逗号+权重。中间以英文分号进行分隔，不要用换行符分隔
            注意你只需给出关键词及权重，其余的什么也不要说
            """
        )
        self.conversation = LLMChain(llm=self.llm, prompt=self.prompt)

    def _test(self):
        test_message = '你好'
        print(self.get_answer(test_message))

    @retry(stop_max_attempt_number=8, wait_fixed=1000, wait_incrementing_increment=2000,
           retry_on_exception=lambda ex: isinstance(ex, ep.ConnectionError))
    def get_answer(self, message):
        time.sleep(3)
        return self.conversation.run(message)


if __name__ == '__main__':
    gpt = GPT()
    data = _load_data()
    articles = np.array([_keep_chinese(text['article']) for text in data])
    years = np.array([int(text['input_date'].split('-')[0]) for text in data])
    tokens = np.array([_cal_token(a, gpt.model, gpt.prompt) for a in articles])

    n_article = articles[tokens <= 4097]
    n_years = years[tokens <= 4097]

    point_years = range(2013, 2024)

    for year in point_years:
        article = n_article[n_years == year]
        if article.size == 0:
            continue

        with open(f'original_data/{year}.txt', 'w') as file:
            for a in article:
                result = gpt.get_answer(a)
                print(result)
                file.write(result + '\n')
