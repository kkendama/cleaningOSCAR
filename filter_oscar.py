import re
import unicodedata
from collections import Counter
import MeCab
from bunkai import Bunkai
from datasets import load_dataset
from multiprocessing import Pool, Manager
from tqdm import tqdm
import jsonlines

dataset = load_dataset("oscar-corpus/OSCAR-2301", use_auth_token=True, language="ja", split="train")

def load_ngwords(directory):
    ngwords = set()
    files = ['adult_keywords_en.txt', 'adult_keywords_ja.txt', 'advertisement_keywords_ja.txt',
             'discrimination_keywords_ja.txt', 'header_footer_keywords_ja.txt', 'violence_keywords_ja.txt']
    for file in files:
        with open(f"{directory}/{file}", "r", encoding="utf-8") as f:
            words = f.read().splitlines()
            # ひらがな・カタカナ2文字以下の単語を除外
            ngwords.update(word for word in words if not re.fullmatch(r'[\u3040-\u30ff]{1,2}', word))
    return ngwords

def is_valid_text(args):
    text, ngwords = args
    text = unicodedata.normalize('NFKC', text)
    # 文字数 (400 文字未満)
    if len(text) <= 400:
        return False

    # 平仮名の文字の割合 (0.2 未満)
    hiragana_count = sum(1 for char in text if re.match(r'[\u3040-\u309F]', char))
    if hiragana_count / len(text) < 0.2:
        return False

    # カタカナの文字の割合 (0.5 以上)
    katakana_count = sum(1 for char in text if re.match(r'[\u30A0-\u30FF]', char))
    if katakana_count / len(text) >= 0.5:
        return False

    # 日本語の文字 (平仮名，カタカナ，漢字，句読点）の割合 (0.5 未満)
    japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF、。]', text)
    if len(japanese_chars) / len(text) < 0.5:
        return False

    # 他の行と重複する行数 / 全行数（0.30）
    lines = text.split('\n')
    unique_lines = set(lines)
    if len(unique_lines) / len(lines) < 0.7:
        return False

    # 他の段落と重複する段落数 / 全段落数（0.30）
    paragraphs = re.split(r'\n{2,}', text)
    unique_paragraphs = set(paragraphs)
    if len(unique_paragraphs) / len(paragraphs) < 0.7:
        return False

    # 他の行と重複する行に含まれる文字数 / 全文字数 (0.20)
    duplicate_line_chars = sum(len(line) for line in lines if lines.count(line) > 1)
    if duplicate_line_chars / len(text) > 0.2:
        return False

    # 他の段落と重複する段落に含まれる文字数 / 全文字数 (0.20)
    duplicate_paragraph_chars = sum(len(paragraph) for paragraph in paragraphs if paragraphs.count(paragraph) > 1)
    if duplicate_paragraph_chars / len(text) > 0.2:
        return False

    # Bunkaiを使用して文を分割
    #bunkai = Bunkai()
    #sentences = list(bunkai(text))
    sentences = re.split(r'\n{1,}', text)
    # sentenceを「。」「!」「?」「.」で分割
    sentences = [re.split(r'([。！？\.]+)', sentence) for sentence in sentences]
    sentences = [item for sublist in sentences for item in sublist if len(item)>1]

    # 文書中の文の文字数の平均 (20 未満，もしくは90 よりも多い場合)
    avg_sentence_length = sum(len(sentence) for sentence in sentences) / len(sentences)
    if avg_sentence_length < 20 or avg_sentence_length > 90:
        return False

    # 最も長い文の文字数 (200 文字以上)
    max_sentence_length = max(len(sentence) for sentence in sentences)
    if max_sentence_length >= 200:
        return False

    # 文末が省略記号で終わる文の割合 (0.2 以上)
    ellipsis_count = sum(1 for sentence in sentences if sentence.endswith('…') or sentence.endswith('...'))
    if ellipsis_count / len(sentences) >= 0.1:
        return False

    # MeCabで形態素解析を行う
    tagger = MeCab.Tagger()
    words = []
    for sentence in sentences:
        node = tagger.parseToNode(sentence)
        while node:
            words.append(node.surface)
            node = node.next

    # 素のテキストでNGワードの登場回数を数える
    ngword_count = sum(text.count(ngword) for ngword in ngwords)
    if ngword_count / len(words) > 0.005:
        return False

    # n-gramの出現回数の条件
    ngram_conditions = [
        (2, 0.20), (3, 0.18), (4, 0.16), (5, 0.15),
        (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10)
    ]
    for n, threshold in ngram_conditions:
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        counter = Counter(ngrams)
        if n < 5:
            most_common_count = counter.most_common(1)[0][1]
            if most_common_count / len(ngrams) > threshold:
                return False
        else:
            repeated_count = sum(count for ngram, count in counter.items() if count > 1)
            if repeated_count / len(ngrams) > threshold:
                return False

    return True

def process_batch(batch, ngwords, writer):
    with Pool() as pool:
        args_list = [(item, ngwords) for item in batch]
        results = pool.map(is_valid_text, args_list)
    valid_items = [{'text': item} for item, valid in zip(batch, results) if valid]
    writer.write_all(valid_items)

def main():
    ngwords = load_ngwords("/root/projects/pretrain-corpus/HojiChar/hojichar/dict")
    
    batch_size = 32784

    with jsonlines.open('filtered_oscar.jsonl', mode='w') as writer:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            try:
                batch = dataset[i:i+batch_size]['text']
                process_batch(batch, ngwords, writer)
            except:
                print("error")
                pass

if __name__ == '__main__':
    main()
