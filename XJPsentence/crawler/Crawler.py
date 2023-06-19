# -*- coding: utf-8 -*-
# @Time : 2023/5/23 21:22
# @Author : DanYang
# @File : Crawler.py
# @Software : PyCharm
import time

import yaml
import requests
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from pymongo import MongoClient


class Crawler:
    def __init__(self):
        with open('./configuration/config.yml', 'r') as file:
            config_data = yaml.safe_load(file)

        self.params = config_data['params']
        self.json_url = config_data['json_url']
        self.crawl_page = config_data['crawl_page']
        self.article_url = config_data['article_url']

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        }

        client = MongoClient("mongodb://localhost:27017/")
        self.db = client['XJPdata']
        self.collection = self.db['diplomats']

    def get_json(self, page):
        time.sleep(0.5)
        self.params['page'] = page
        query_string = urlencode(self.params)
        url = self.json_url + query_string

        response = requests.get(url, headers=self.headers)
        response.encoding = 'utf-8'
        return response.json()['list']

    def get_detail_data(self, data):
        article_ids = [i['article_id'] for i in data]
        for article_id, d in zip(article_ids, data):
            time.sleep(0.6)
            url = self.article_url.format(id=article_id)
            response = requests.get(url, headers=self.headers)
            response.encoding = 'utf-8'
            content = BeautifulSoup(response.text, 'lxml')
            ps = content.select('div.d2txt_con p')
            article = ''
            for p in ps:
                article = article + p.text
            d['article'] = article
            self.save_data(d)

    def save_data(self, data):
        self.collection.insert_one(data)

    def scheduling(self, page):
        data = self.get_json(page)
        self.get_detail_data(data)


if __name__ == '__main__':
    c = Crawler()
