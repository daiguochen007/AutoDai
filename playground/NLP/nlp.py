# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 00:12:07 2021

@author: Dai
"""

# import nltk
# nltk.download()

from nltk.book import text1

# find given word
text1.concordance("crazy")
text1.similar("crazy")

text1.dispersion_plot(["this", "crazy", "freedom", "duties", "lovely", "you"])


