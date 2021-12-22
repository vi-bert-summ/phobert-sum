import streamlit as st
import os
import torch
import nltk
import urllib.request
from models.ext_bert_summ_pylight import ExtBertSummPylight
from newspaper import Article
from vncorenlp import VnCoreNLP
import json
from ext_sum import summarize as ext_summarize


def main():
    st.markdown("<h1 style='text-align: center;'>Tóm tắt văn bản Tiếng Việt</h1>", unsafe_allow_html=True)

    # Download model
    if not os.path.exists('checkpoints/phobert_ext.pt'):
        download_model()
        
    # Load model
    model = load_model()

    # Input
    input_type = st.radio("Input Type: ", ["URL", "Raw Text"])
    st.markdown("<h3 style='text-align: center;'>Input</h3>", unsafe_allow_html=True)
    
    # Segmenter define
    VNCORENLP_PATH = "VnCoreNLP-1.1.1\VnCoreNLP-1.1.1.jar"
    segmenter = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx500m') 
    
    if input_type == "Raw Text":
        with open("raw_data/input.txt", "r", encoding='utf-8') as f:
            sample_text = f.read()
        text = st.text_area("", sample_text, 200)
        item = tokenize(text, segmenter)
        
    else:
        url = st.text_input("", "https://dantri.com.vn/suc-khoe/ha-noi-them-1704-ca-covid19-485-ca-cong-dong-20211221174836611.htm")
        st.markdown(f"[*Read Original News*]({url})")
        text = crawl_url(url)
        item = tokenize(text, segmenter)

    input_fp = "raw_data/input.json"
    # with open(input_fp, 'w', encoding="utf-8") as file:
    #     file.write(text)

    with open(input_fp, 'w', encoding='utf-8') as fo:
        json.dump(item, fo, indent=4, ensure_ascii=False)

    # Summarize
    sum_type = st.radio("Mode: ", ["Extractive", "Abstractive"])
    result_fp = 'results/summary.txt'
    if sum_type == "Extractive":
        summary = ext_summarize(input_fp, result_fp, model)
    else:
        summary = ""
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)


def download_model():
    nltk.download('popular')
    url = 'https://docs.google.com/uc?export=download&id=1kjy_yDO7gAbzEWGi5CfwXfmfCNb__V0i' 
    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading checkpoint...")
        progress_bar = st.progress(0)
        with open('checkpoints/phobert_ext.pt', 'wb') as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading checkpoint... (%6.2f/%6.2f MB)" %
                        (counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


@st.cache(suppress_st_warning=True)
def load_model():
    # checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
    model = ExtBertSummPylight()
    return model


def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def tokenize(texts, segmenter):
    texts = texts.split('\n\n')
    tokenized_texts = []
    for text in texts:
        tokenized_texts.append(segmenter.tokenize(text)[0])
    item = {}
    item['src'] = tokenized_texts
    return item
        
if __name__ == "__main__":
    main()
    