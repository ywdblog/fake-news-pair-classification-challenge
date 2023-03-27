# -*- coding: utf-8 -*

from keras.utils import plot_model
import keras
import numpy
import pandas as pd
import tensorflow as tf
import jieba.posseg as pseg
import sys
from sklearn.model_selection import train_test_split


# 建立孿生 LSTM 架構（Siamese LSTM）
from keras import Input
from keras.layers import Embedding,  LSTM, concatenate, Dense
from keras.models import Model


VALIDATION_RATIO = 0.1


def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([
        word for word, flag in words if flag != 'x'])


# 在語料庫裡有多少詞彙
MAX_NUM_WORDS = 10000
# 一個標題最長有幾個詞彙
MAX_SEQUENCE_LENGTH = 20


# 一個詞向量的維度
NUM_EMBEDDING_DIM = 256
# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 128
# 基本參數設置，有幾個分類
NUM_CLASSES = 3

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_NUM_WORDS
)

label_to_index = {
    'unrelated': 0,
    'agreed': 1,
    'disagreed': 2
}


TRAIN_CSV_PATH = "C:\\Users\\ywdblog\\Desktop\\xwj\\pyproject\\example\\fake-news-pair-classification-challenge\\train-test2.csv"


TEST_CSV_PATH = "C:\\Users\\ywdblog\\Desktop\\xwj\\pyproject\\example\\fake-news-pair-classification-challenge\\test.csv"

train = pd.read_csv(
    TRAIN_CSV_PATH, index_col=0)

# print(train.label[0:5])


y_train = train.label.apply(lambda x: label_to_index[x])  # 将标签转换为数字
y_train = numpy.asarray(y_train).astype('float32')

y_train = tf.keras.utils.to_categorical(y_train)  # one-hot 编码

#
print(y_train[:5])

train['title1_tokenized'] = train.loc[:, 'title1_zh'].apply(jieba_tokenizer)
train['title2_tokenized'] = train.loc[:, 'title2_zh'].apply(jieba_tokenizer)


corpus_x1 = train.title1_tokenized
corpus_x2 = train.title2_tokenized
corpus = pd.concat([
    corpus_x1, corpus_x2])
tokenizer.fit_on_texts(corpus)  # 生成词汇表（word_index）（词典
print(corpus.shape)

# debug
# corpus.iloc[:5] 选择了 corpus 数据集中的前5行，而 ['title'] 将结果限制为只包含 title 列
#print(pd.DataFrame(corpus.iloc[:5], columns=['title1_tokenized']))

x1_train = tokenizer.texts_to_sequences(corpus_x1)  # 将文本转换为序列
x2_train = tokenizer.texts_to_sequences(corpus_x2)

x1_train = tf.keras.preprocessing.sequence.pad_sequences(
    x1_train, maxlen=MAX_SEQUENCE_LENGTH)  # 将序列转换为张量 使得每个序列的长度都是相同的
x2_train = tf.keras.preprocessing.sequence.pad_sequences(
    x2_train, maxlen=MAX_SEQUENCE_LENGTH)


print(len(x1_train))
print(x1_train[:1])
print(x1_train[0])

# for seq in x1_train[:1]:
#     print([tokenizer.index_word[idx] for idx in seq])

# 序列长度不一样
for seq in x1_train[:10]:
    print(len(seq), seq[:5], ' ...')

x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
    x1_train, x2_train, y_train,
    test_size=VALIDATION_RATIO,
    # random_state=RANDOM_STATE
)

print("Training Set")
print("-" * 10)
print(f"x1_train: {x1_train.shape}")
print(f"x2_train: {x2_train.shape}")
print(f"y_train : {y_train.shape}")

print("-" * 10)
print(f"x1_val:   {x1_val.shape}")
print(f"x2_val:   {x2_val.shape}")
print(f"y_val :   {y_val.shape}")
print("-" * 10)
print("Test Set")
# print(train.label[:5])


for i, seq in enumerate(x1_train[:5]):
    print(f"新闻标题 {i + 1}: ")
    print(seq)
    print()

for i, seq in enumerate(x1_train[:5]):
    print(f"新闻标题 {i + 1}: ")
    print([tokenizer.index_word.get(idx, '') for idx in seq])
    print()


# 分別定義 2 個新聞標題 A & B 為模型輸入
# 兩個標題都是一個長度為 20 的數字序列
top_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ),
    dtype='int32')
bm_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ),
    dtype='int32')

# 詞嵌入層
# 經過詞嵌入層的轉換，兩個新聞標題都變成
# 一個詞向量的序列，而每個詞向量的維度
# 為 256
embedding_layer = Embedding(
    MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
top_embedded = embedding_layer(
    top_input)
bm_embedded = embedding_layer(
    bm_input)

# LSTM 層
# 兩個新聞標題經過此層後
# 為一個 128 維度向量
shared_lstm = LSTM(NUM_LSTM_UNITS)
top_output = shared_lstm(top_embedded)
bm_output = shared_lstm(bm_embedded)

# 串接層將兩個新聞標題的結果串接單一向量
# 方便跟全連結層相連
merged = concatenate(
    [top_output, bm_output],
    axis=-1)

# 全連接層搭配 Softmax Activation
# 可以回傳 3 個成對標題
# 屬於各類別的可能機率
dense = Dense(
    units=NUM_CLASSES,
    activation='softmax')
predictions = dense(merged)

# 我們的模型就是將數字序列的輸入，轉換
# 成 3 個分類的機率的所有步驟 / 層的總和
model = Model(
    inputs=[top_input, bm_input],
    outputs=predictions)

# # pip install pydot
# plot_model(
#     model,
#     to_file='model.png',
#     show_shapes=True,
#     show_layer_names=False,
#     rankdir='LR')


# 查看每一層的參數量以及輸出的張量（Tensor）長相
model.summary()


# 定义模型的損失函數
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# 決定一次要放多少成對標題給模型訓練
BATCH_SIZE = 512

# 決定模型要看整個訓練資料集幾遍
NUM_EPOCHS = 10


# # 實際訓練模型
# history = model.fit(
#     # 輸入是兩個長度為 20 的數字序列
#     x=[x1_train, x2_train],
#     y=y_train,
#     batch_size=BATCH_SIZE,
#     epochs=NUM_EPOCHS,
#     # 每個 epoch 完後計算驗證資料集
#     # 上的 Loss 以及準確度
#     validation_data=(
#         [x1_val, x2_val],
#         y_val
#     ),
#     # 每個 epoch 隨機調整訓練資料集
#     # 裡頭的數據以讓訓練過程更穩定
#     shuffle=True
# )
# print("end1")
#########################
test = pd.read_csv(TEST_CSV_PATH, index_col=0)


test.dropna(axis=0, subset=['title1_zh', 'title2_zh'], inplace=True)


# 以下步驟分別對新聞標題 A、B　進行
# 文本斷詞 / Word Segmentation
test['title1_tokenized'] = test.loc[:, 'title1_zh'].apply(jieba_tokenizer)
test['title2_tokenized'] = test.loc[:, 'title2_zh'].apply(jieba_tokenizer)


# 將詞彙序列轉為索引數字的序列
x1_test = tokenizer.texts_to_sequences(test.title1_tokenized)
x2_test = tokenizer.texts_to_sequences(test.title2_tokenized)

# 為數字序列加入 zero padding
x1_test = tf.keras.preprocessing.sequence.pad_sequences(
    x1_test, maxlen=MAX_SEQUENCE_LENGTH)

x2_test = tf.keras.preprocessing.sequence.pad_sequences(
    x2_test,      maxlen=MAX_SEQUENCE_LENGTH)

#  利用已訓練的模型做預測
predictions = model.predict([x1_test, x2_test])

# 预测结果
print(predictions[:5])

print("end2")
