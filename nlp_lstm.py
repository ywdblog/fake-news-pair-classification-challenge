# -*- coding: utf-8 -*

# conda install -c conda-forge jieba
# conda install -c conda-forge keras
# conda install -c conda-forge tensorflow
# conda install -c conda-forge pandas
# conda install -c conda-forge numpy
# conda install -c conda-forge matplotlib
# conda install -c conda-forge scikit-learn
# conda install -c conda-forge jieba 

from keras.utils import plot_model
import keras
import numpy
import pandas as pd
import tensorflow as tf
import jieba.posseg as pseg
import sys
from sklearn.model_selection import train_test_split
from keras import Input
from keras.layers import Embedding,  LSTM, concatenate, Dense
from keras.models import Model
from tensorflow.keras.models import load_model

env = "load"
if len(sys.argv) > 1:
    env = sys.argv[1]

# jieba 分词
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([
        word for word, flag in words if flag != 'x'])
 
TRAIN_CSV_PATH = "train-small.csv"
TEST_CSV_PATH = "test-small.csv"
MODEL_PATH = "model.h5"

 
label_to_index = {
    'unrelated': 0,
    'agreed': 1,
    'disagreed': 2
}

train = pd.read_csv(
    TRAIN_CSV_PATH, index_col=0)

# 选择需要的列，看看数据集的前3行
cols = ['title1_zh', 
        'title2_zh', 
        'label']
train = train.loc[:, cols]
train.head(3)

print(train.head(3))
 
y_train = train.label.apply(lambda x: label_to_index[x])  # 将标签转换为数字
y_train = numpy.asarray(y_train).astype('float32')
y_train = tf.keras.utils.to_categorical(y_train)  # one-hot 编码
print(y_train[:5])

# 在语料库有多少个词条
MAX_NUM_WORDS = 10000

# keras集成功能：将文本切割为单词，并构建索引词典，最终将文本转换为序列
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_NUM_WORDS
)

# jieba 分词
train['title1_tokenized'] = train.loc[:, 'title1_zh'].apply(jieba_tokenizer)
train['title2_tokenized'] = train.loc[:, 'title2_zh'].apply(jieba_tokenizer)
 
corpus_x1 = train.title1_tokenized
corpus_x2 = train.title2_tokenized
# corpus 也叫语料库，是指用于训练模型的数据集
corpus = pd.concat([
    corpus_x1, corpus_x2])
 

# corpus.iloc[:5] 选择corpus数据集中的前5行，而 ['title'] 将结果限制为只包含title列
#print(pd.DataFrame(corpus.iloc[:5], columns=['title1_tokenized']))

# 生成词汇表（word_index）词典的key是单词，value是单词的索引
tokenizer.fit_on_texts(corpus)   
print(corpus.shape)

# 一个标题最长有多少个词条
MAX_SEQUENCE_LENGTH = 20
 
# 将文本转换为序列
x1_train = tokenizer.texts_to_sequences(corpus_x1)   
x2_train = tokenizer.texts_to_sequences(corpus_x2)

# 将序列转换为张量，使得每个序列的长度都是相同的
x1_train = tf.keras.preprocessing.sequence.pad_sequences(
    x1_train, maxlen=MAX_SEQUENCE_LENGTH)   
x2_train = tf.keras.preprocessing.sequence.pad_sequences(
    x2_train, maxlen=MAX_SEQUENCE_LENGTH)

print(len(x1_train)) # 10000 
print(x1_train[:1]) 
print(x1_train[0]) #

# 序列长度不一样
for seq in x1_train[:10]:
    print(len(seq), seq[:5], ' ...')


 
# train_test_split 函数将数据集分为训练集和验证集
VALIDATION_RATIO = 0.1 
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
print("Valid Set")
# print(train.label[:5])

# enumerate 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标 
for i, seq in enumerate(x1_train[:5]):
    print(f"新闻标题 {i + 1}: ")
    print(seq)
    print()

for i, seq in enumerate(x1_train[:5]):
    print(f"新闻标题 {i + 1}: ")
    print([tokenizer.index_word.get(idx, '') for idx in seq])
    print()

# 需要解决的问题：如何将一个词条转换为一个N维向量？且这个向量能够表示这个词条的语义信息（Semantic）？
# 如果能够把语料库（Corpus）里头的每个词汇都表示为一个有意义的向量，神经网络就能找到潜藏在大量词条中的语义关系，从而提升NLP精确度
# 大部分情况下，不需要手动设置词汇的词向量，完全可以随机初始化词向量，然后利用反向传播算法自动学到NLP中的词向量，这种技术也叫词嵌入（word embedding）


# 一个词向量的维度
NUM_EMBEDDING_DIM = 256
# LSTM 输出的向量维度
NUM_LSTM_UNITS = 128
# 分类数量
NUM_CLASSES = 3

# Input 函数用于实例化一个 Keras tensor
# top_input vs bm_input 两个新闻标题 A & B 作为模型输入 
top_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ),
    dtype='int32')
bm_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ),
    dtype='int32')

# Embedding 函数将正整数（下标）转换为具有固定大小的向量
# 经过 Embedding 层的转换，两个新闻标题都变成了一个词向量的序列，而每个词向量的维度为 256
embedding_layer = Embedding(
    MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
top_embedded = embedding_layer(
    top_input)
bm_embedded = embedding_layer(
    bm_input)
# 有了Embedding词向量，就可以将其丢给LSTM 层，有了两个标题党词向量，接下去就是处理神经网络架构

# LSTM 层
# 两个新闻标题经过此层后，为一个 128 维度向量
# A与B标题共享一个LSTM
shared_lstm = LSTM(NUM_LSTM_UNITS)
top_output = shared_lstm(top_embedded)
bm_output = shared_lstm(bm_embedded)

# 串接层将两个新闻标题的结果串接成一个向量，方便跟全连接层相连
merged = concatenate(
    [top_output, bm_output],
    axis=-1)

# Dense 函数为全连接层，搭配 Softmax Activation 可以返回3个成对标题属于各类别的可能概率
dense = Dense(
    units=NUM_CLASSES,
    activation='softmax')
predictions = dense(merged)

# Softmax 函数将一个K维的实数向量转换为一个K维的实数概率向量，使得向量中的每个元素都处于0到1之间，并且所有元素的和为1

# 模型就是将数字序列的输入，转换成 3 个分类的机率的所有步骤/层的总和
model = Model(
    inputs=[top_input, bm_input],
    outputs=predictions)

# 画出模型结构图
plot_model(
    model,
    to_file='model.png',
    show_shapes=True,
    show_layer_names=False,
    rankdir='LR')

# summary 函数可以打印出模型的结构
model.summary()

# 编译模型
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

BATCH_SIZE = 512
NUM_EPOCHS = 10

if env == 'fit':
    # 训练模型
    print("Training Start")
    history = model.fit(
        x=[x1_train, x2_train],
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(
            [x1_val, x2_val],
            y_val
        ),
        shuffle=True
    )
    model.save(MODEL_PATH)

    print("Training Finished")

print("Predicting")

test = pd.read_csv(TEST_CSV_PATH, index_col=0)
test.dropna(axis=0, subset=['title1_zh', 'title2_zh'], inplace=True)

# 文本分词
# word segmentation 
test['title1_tokenized'] = test.loc[:, 'title1_zh'].apply(jieba_tokenizer)
test['title2_tokenized'] = test.loc[:, 'title2_zh'].apply(jieba_tokenizer)

# 将词汇序列转换为索引数字的序列
x1_test = tokenizer.texts_to_sequences(test.title1_tokenized)
x2_test = tokenizer.texts_to_sequences(test.title2_tokenized)
 
# 将序列填充为相同长度
x1_test = tf.keras.preprocessing.sequence.pad_sequences(
    x1_test, maxlen=MAX_SEQUENCE_LENGTH)

x2_test = tf.keras.preprocessing.sequence.pad_sequences(
    x2_test,      maxlen=MAX_SEQUENCE_LENGTH)

if env != 'fit':
    print("Loading Model")
    loadmodel = load_model(MODEL_PATH)
    predictions = loadmodel.predict([x1_test, x2_test])
else :
    predictions = model.predict([x1_test, x2_test])

# 预测结果
print(predictions[:5])

index_to_label = {v: k for k, v in label_to_index.items()}

test['Category'] = [index_to_label[idx] for idx in numpy.argmax(predictions, axis=1)]

submission = test \
    .loc[:, ['Category']] \
    .reset_index()

submission.columns = ['Id', 'Category']
submission.head()