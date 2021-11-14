# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 19:31:10 2021

@author: deadp
"""

import glassdoor_scraper as gs
import pandas as pd

path = "D:/Projects/Data Science Salary Predictor/chromedriver"

df = gs.get_jobs('data scientist', 15, False, path, 5)