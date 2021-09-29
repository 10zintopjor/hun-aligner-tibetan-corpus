import re
from botok.tokenizers.wordtokenizer import WordTokenizer
from botok.tokenizers.sentencetokenizer import sentence_tokenizer
from pathlib import Path
import spacy

def get_tokens(text):
    wt = WordTokenizer()
    tokens = wt.tokenize(text, split_affixes=False)
    return tokens

def get_sentences(text):
    tokens = get_tokens(text)
    sentences = sentence_tokenizer(tokens)
    return sentences

def serialize_sentence(sentence):
    sentence_content = ''
    for token in sentence:
        sentence_content += f'{token.text} '
    return sentence_content

def preprocess_bo_text(text):
    new_bo_text = '#\n'
    text = text.replace('\n', ' ')
    sentences = get_sentences(text)
    for sent_len, sentence in sentences:
        new_bo_text += serialize_sentence(sentence) + '\n'
    new_bo_text += '#'
    return new_bo_text

def preprocess_en_text(text):
    new_eng_text = '#\n'
    text = text.replace('\n', '')
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    for sentence in doc.sents:
        new_eng_text += str(sentence) + '\n'
    new_eng_text += '#'
    return new_eng_text

def preprocess_corpus(bo_text_path, en_text_path):
    bo_text = Path(bo_text_path).read_text(encoding='utf-8')
    en_text = Path(en_text_path).read_text(encoding='utf-8')
    preprocess_bo = preprocess_bo_text(bo_text)
    preprocess_en = preprocess_en_text(en_text)
    post_en_text_path = f'{en_text_path[:-4]}_post.txt'
    post_bo_text_path = f'{bo_text_path[:-4]}_post.txt'
    Path(post_bo_text_path).write_text(preprocess_bo, encoding='utf-8')
    Path(post_en_text_path).write_text(preprocess_en, encoding='utf-8')
    return [post_bo_text_path, post_en_text_path]


