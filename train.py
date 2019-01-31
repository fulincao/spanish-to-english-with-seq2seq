import model
import config
import tensorflow as tf
import numpy as np
import time
import os
import utils
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s -- %(filename)s[line:%(lineno)d]  -- %(levelname)s  --  '
                                               '%(message)s')


def train():
    # 开启tensorflow 动态图功能
    tf.enable_eager_execution()
    # generate dataset
    num_examples = 3000
    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar = utils.load_dataset(num_examples)
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)
    logging.info('验证集和数据集大小' + str(len(input_tensor_train)) + ' ' + str(len(input_tensor_val)))

    # 创建tf.data.dataset数据集并且设定部分参数
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = config.BATCH_SIZE
    N_BATCH = BUFFER_SIZE / BATCH_SIZE
    embedding_dim = config.EMBEDDING_DIM
    units = config.UNITS
    vocab_inp_size = config.VOCAB_INP_SIZE
    vocab_tar_size = config.VOCAB_TAR_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # 解码器，编码器
    encoder = model.Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = model.Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    # 优化器
    optimizer = tf.train.AdamOptimizer()

    # 损失函数
    def loss_function(real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    # checkpoints(object-based saving)
    checkpoint_dir = config.CHECK_POINT_DIR
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    # 训练
    EPOCHS = config.EPOCHS
    for epoch in range(EPOCHS):
        start = time.time()
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)
                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    loss += loss_function(targ[:, t], predictions)
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))
            total_loss += batch_loss
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    train()
