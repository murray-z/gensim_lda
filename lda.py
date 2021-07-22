# -*- coding: utf-8 -*-

import re
import jieba
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import cossim


# jieba加载自定义词典和停用词
user_dict = "./data/user_dict.txt"
stopwords_path = "./data/stop_words.txt"
jieba.load_userdict(user_dict)
stopwords = [word.strip() for word in open(stopwords_path, encoding="utf-8").readlines()]


def generator_corpus(data_path):
    corpus = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = re.sub("[\s\r\n]+", "", line)
            lis = jieba.lcut(line)
            corpus.append([word for word in lis if word not in stopwords])
    return corpus


def train_lda(data_path, num_topics=10,
              lda_model_save_path="./lda_model/lda.model",
              dictionary_save_path="./lda_model/dic.pkl"):
    """
    训练模型
    :param data_path: 数据地址，每行为一个文本
    :param num_topics: 主题数量
    :param lda_model_save_path: lda模型保存路径
    :param dictionary_save_path: dic保存路径
    :return:
    """
    # 生成训练模型所需要的语料:[[words]]
    corpus = generator_corpus(data_path)
    # 按照字典对语料进行转换
    dictionary = Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]
    # 开始训练
    lda = LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
    # 评估模型
    perplexity = lda.log_perplexity(corpus)
    print("perplexity: {}".format(perplexity))
    # 保存模型
    lda.save(lda_model_save_path)
    dictionary.save(dictionary_save_path)

def show_topics(lda_model_path, dic_path, num_topics=-1, num_words=10):
    """
    打印主题及其关键词分布
    :param lda_model_path: 模型地址
    :param dic_path: 字典地址
    :param num_topics: 打印主题数目， -1表示所有主题
    :param num_words: 打印每个主题关键词数量
    :return:
    """
    lda = LdaModel.load(lda_model_path)
    dic = Dictionary.load(dic_path)
    id2token = {str(id): token for token, id in dic.token2id.items()}
    res = lda.show_topics(num_topics, num_words, formatted=False)
    for topic_id, words_weight in res:
        print("topic_id: {}".format(topic_id))
        for word_id, weight in words_weight:
            print("{}:{}".format(id2token[word_id], weight))
        print("\n")


def update_lda(row_model_path, new_corpus, new_model_save_path):
    """采用新的语料继续训练模型"""
    old_model = LdaModel.load(row_model_path)
    old_model.update(new_corpus)
    old_model.save(new_model_save_path)


def trans_text_to_vec(text, dictionary):
    return dictionary.doc2bow([word for word in jieba.lcut(text) if word not in stopwords])


def calculate_texts_similarity(text1, text2,
                               model_path="./lda_model/lda.model",
                               dic_path="./lda_model/dic.pkl"):
    """
    计算两个文本的主题分布相似性
    :param text1:
    :param text2:
    :param model_path: 模型地址
    :param dic_path: 字典地址
    :return:
    """
    # 加载模型和词典
    lda = LdaModel.load(model_path)
    dic = Dictionary.load(dic_path)

    # 文本转成词向量
    vec1 = trans_text_to_vec(text1, dic)
    vec2 = trans_text_to_vec(text2, dic)

    # lda计算主题分布
    topic1 = lda[vec1]
    topic2 = lda[vec2]
    print(len(topic1), topic1)
    print(len(topic2), topic2)
    # 计算主题分布相似性
    simi = cossim(topic1, topic2)
    return simi


if __name__ == '__main__':
    outside_path = "./data/texts.txt"
    train_lda(outside_path, )

    text1 = """"""
    text2 = """""
    simi = calculate_texts_similarity(text1, text2)
    print(simi)

    # show_topics("./lda_model/lda.model", "./lda_model/dic.pkl")
