import numpy as np
import pandas as pd
import config as cfg

import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

def auto_interacting(embed_map, d=cfg.d, H=cfg.H):
    k = embed_map.shape[-1]

    attention_heads = []
    w_query = []
    w_key = []
    w_value = []

    for i in range(H):
        w_query.append(tf.Variable(tf.random.truncated_normal(shape=(k, d)), name='query_' + str(i)))
        w_key.append(tf.Variable(tf.random.truncated_normal(shape=(k, d)), name='key_' + str(i)))
        w_value.append(tf.Variable(tf.random.truncated_normal(shape=(k, d)),name='value_' + str(i)))

    for i in range(H):
        embed_query = tf.matmul(embed_map, w_query[i])
        embed_key = tf.matmul(embed_map, w_key[i])
        embed_value = tf.matmul(embed_map, w_value[i])

        attention = tf.nn.softmax(tf.matmul(embed_query, tf.transpose(embed_key, [0, 2, 1])))

        attention_output = tf.matmul(attention, embed_value)

        attention_heads.append(attention_output)
    
    attention_outputs = Concatenate(axis=-1)(attention_heads)

    w_res = tf.Variable(tf.random.truncated_normal(shape=(k, d * H)), name='w_res')
    output = Activation('relu')(attention_outputs + tf.matmul(embed_map, w_res))

    return output

def build_autoint(x0, n_layers=cfg.n_layers):
    xl = x0
    for i in range(n_layers):
        xl = auto_interacting(xl)
    return xl

def build_model(dense_features, sparse_features, total_data):
    print('Building autoint model...')
    k = cfg.k

    dense_inputs = []
    for f in dense_features:
        _input = Input([1], name=f)
        dense_inputs.append(_input)

    dense_kd_embed = []
    for i, _input in enumerate(dense_inputs):
        f = dense_features[i]
        embed = tf.Variable(tf.random.truncated_normal(shape=(1, k), stddev=0.01), name=f)
        scaled_embed = tf.expand_dims(_input * embed, axis=1)
        dense_kd_embed.append(scaled_embed)

    sparse_inputs = []
    for f in sparse_features:
        _input = Input([1], name=f)
        sparse_inputs.append(_input)

    sparse_kd_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = sparse_features[i]
        voc_size = total_data[f].nunique()
        _embed = Embedding(voc_size+1, k, embeddings_regularizer=tf.keras.regularizers.l2(0.5))(_input)
        sparse_kd_embed.append(_embed)

    input_embeds = dense_kd_embed + sparse_kd_embed
    embed_map = Concatenate(axis=1)(input_embeds)

    autoint_layer = build_autoint(embed_map, n_layers=cfg.n_layers)

    autoint_layer = Flatten()(autoint_layer)

    output_layer = Dense(1, activation='sigmoid')(autoint_layer)

    model = Model(dense_inputs + sparse_inputs, output_layer)

    model.compile(optimizer=cfg.optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')])

    print('done')

    return model

