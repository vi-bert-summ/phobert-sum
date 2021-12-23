import os
import numpy as np
import nltk
import torch
import urllib.request
import streamlit as st

import utils

from newspaper import Article
import os
import time
import glob
import datasets
import pandas as pd
import transformers
import concurrent.futures
from typing import Optional

import utils
import parameters

# from datasets import *
from vncorenlp import VnCoreNLP
from transformers import EncoderDecoderModel
from transformers import AutoTokenizer


args = parameters.get_args()


def main():
    start = time.time()
    bg_img = '''
    <style>
    body {
    background: url("https://img1.picmix.com/output/stamp/normal/3/7/8/5/1395873_5d4a6.gif"),
                url("https://img1.picmix.com/output/stamp/normal/3/7/8/5/1395873_5d4a6.gif");
    background-repeat: no-repeat no-repeat;
    background-position: 0% -40%, 100% -40%
    }
    </style>
    '''

    st.markdown(bg_img, unsafe_allow_html=True)
    # st.markdown(bg_img_right, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>Tóm Tắt Văn Bản Tiếng Việt</h1>", unsafe_allow_html=True)

    # Download model
    if not os.path.exists('checkpoints/mobilebert_ext.pt'):
        # download_model()
        pass

    # Load model
    # model = load_model()

    # Input
    input_type = st.radio("Định dạng đầu vào: ", ["URL", "Đoạn văn"])
    st.markdown("<h3 style='text-align: center;'>Đầu vào</h3>", unsafe_allow_html=True)

    if input_type == "Raw Text":
        text = st.text_input("")
    else:
        url = st.text_input("", "https://www.cnn.com/2020/05/29/tech/facebook-violence-trump/index.html")
        st.markdown(f"[*Nguồn văn bản*]({url})")
        text = crawl_url(url)

    text = text.replace('_', ' ')
    # Summarize
    sum_level = st.radio("Kiểu tóm tắt: ", ["Trích rút", "Tóm lược"])
    result = st.button('Tóm tắt')
    summary = ''
    time_consuming = ''
    if result: 
        rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 

        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        summary = utils.bertsum(text, tokenizer, rdrsegmenter, args.device, args.checkpoint)
        time_consuming = int(time.time() - start)
        time_consuming = str(time_consuming) + 's'

    summary = summary.replace('_', ' ')
    # max_length = 3 if sum_level == "Short" else 5
    # result_fp = 'results/summary.txt'
    # summary = summarize(input_fp, result_fp, model, max_length=max_length)
    st.markdown("<h3 style='text-align: center;'>Nội dung gốc</h3>", unsafe_allow_html=True)
    st.markdown(f"<p align='justify'>{text}</p>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Văn bản tóm tắt {}</h3>".format(time_consuming), unsafe_allow_html=True)
    st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)

@st.cache(suppress_st_warning=True)
def load_model():
    model = EncoderDecoderModel.from_pretrained(args.checkpoint)
    model.to(args.device)
    return model


def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


if __name__ == "__main__":
    main()


