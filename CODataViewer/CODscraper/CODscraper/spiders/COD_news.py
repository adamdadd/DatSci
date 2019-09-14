# -*- coding: utf-8 -*-
import scrapy


class CodNewsSpider(scrapy.Spider):
    name = 'COD_news'
    allowed_domains = ['opendata.cern.ch/record/201']
    start_urls = ['http://opendata.cern.ch/record/201']

    def parse(self, response):
        title = response.css('.d-inline::text').extract()
            desc = response.css(' p::text').extract()

        for item in zip(title, desc):
            scraped_info = {
                'Title' : item[0],
                'Description' : item[1]
            }

            yield scraped_info
