import tensorflow as tf
import unicodedata
import re

# path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip',extract=True, cache_dir='./data')
path_to_file = "./data/spa-eng/spa.txt"


# 第一个参数指定字符串标准化的方式。 NFC表示字符应该是整体组成(比如可能的话就使用单一编码)，
# 而NFD表示字符应该分解为多个组合字符表示。unicodedata.category(chr) 把一个字符返回它在UNICODE里分类的类型
# unicode file 2 ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# 预处理
def preprocess_sentence(w):
    # 转小写换编码
    w = unicode_to_ascii(w.lower().strip())
    # 替换 \1 代表匹配到的第一个内容 \1_ 替换分隔符并加了个空格
    #  creating a space between a word and the punctuation following it
    #  eg: "he is a boy." => "he is a boy ，
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" ]+', " ", w)
    # rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
    w = w.rstrip().strip()
    # 加开始和结束字符
    w = '<start> ' + w + ' <end>'
    return w


# 从文件中前num_examples创建数据集
def create_dataset(path, num_examples):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return word_pairs


# 通过lang创建word2id和id2word和vocab表
class LanguageIndex:
    def __init__(self, lang, sep=' '):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.sep = sep
        self.create_index()

    def create_index(self):
        if self.sep != '':
            for phrase in self.lang:
                self.vocab.update(phrase.split(self.sep))
        else:
            for phrase in self.lang:
                self.vocab.update(x for x in phrase)
        self.vocab = sorted(self.vocab)
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1
        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset(num_examples):
    # 创建输出
    pairs = create_dataset(path_to_file, num_examples)
    # 输入输出词汇表和相应的映射
    inp_lang = LanguageIndex(sp for en, sp in pairs)
    targ_lang = LanguageIndex(en for en, sp in pairs)
    # Spanish, English sentence in id
    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    # 补0到所有向量到maxlen长度
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length_inp, padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length_tar, padding='post')
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar


def xml_parse():
    with open('./data/sohu/news_tensite_xml.smarty.xml', encoding='utf-8') as f:
        cont = f.read().replace('\n', '').strip()
        mx = re.findall('<doc>.*?<contenttitle>(.*?)</contenttitle>.*?<content>(.*?)</content>.*?</doc>', cont)
        for item in mx:
            title = item[0].strip().replace(' ', '，')
            content = re.sub(r'\\u[0-9A-Za-z]{4}', '', item[1].strip()).replace(' ', '').strip()
            if title == '' or content == '':
                continue
            print(title, content)
        print(mx)


if __name__ == '__main__':
    xml_parse()
