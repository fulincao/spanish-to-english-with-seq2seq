import utils
import config
import tensorflow as tf
import time
import numpy as np
import model
import os
import gc


def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = utils.preprocess_sentence(sentence)
    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.idx2word[predicted_id] + ' '
        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence, attention_plot


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp,max_length_targ)
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))


if __name__ == '__main__':
    num_examples = 3000
    # 开启tensorflow 动态图功能
    tf.enable_eager_execution()
    # 解码器，编码器
    vocab_inp_size, vocab_tar_size = config.VOCAB_INP_SIZE, config.VOCAB_TAR_SIZE
    embedding_dim, units, BATCH_SIZE = config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE
    _, __, inp_lang, targ_lang, max_length_inp, max_length_tar = utils.load_dataset(num_examples)
    del _, __
    gc.collect()
    encoder = model.Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = model.Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    optimizer = tf.train.AdamOptimizer()


    # checkpoints(object-based saving)
    checkpoint_dir = config.CHECK_POINT_DIR
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    last = time.time()
    translate(u'hace mucho frio aqui.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_tar)
    print(time.time() - last)
