#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .seq_op import *
from .att_op import *

def build_raw_aspect(embed, 
                hdim=10,
                num_aspect=5,
                filter_size=3,
                initializer=None, padding='SAME',
                dropout=None, name='', reuse=None):
    """ Builds a Aspect Neural Encoder and returns hidden outputs instead

    Args:
        embed: `tensor` input embedding of shape bsz x time_steps x dim
        hdim: `int` dimension of aspect embedding
        num_aspect: `int` number of aspects k
        filter_sizes: `int` - filter size for convolutiontal attention c
        initializer: tensorflow initializer
        dropout: tensorfow dropout placeholder
        reuse: to reuse weights or not

    Returns: - no pooling
        outputs: `tensor` output embedding of shape
            [bsz x time_steps x hdim]

    """

    embed_expanded = tf.expand_dims(embed, 1)
    embed_expanded = tf.tile(embed_expanded, [1, num_aspect, 1, 1]) # B x K x n x d

    #batch_size = embed_expanded.get_shape().as_list()[0]
    batch_size = tf.shape(embed_expanded)[0] # B
    dim = embed_expanded.get_shape().as_list()[3] # d

    var_name = "raw_aspect_projection_layer_{}".format(name)
    with tf.variable_scope(var_name, reuse=reuse) as scope:
        filter_shape = [num_aspect, dim, hdim] # K x d x h1
        Wa = tf.get_variable("weights", filter_shape, initializer=initializer)
    Wa_expanded = tf.expand_dims(Wa, 0)
    Wa_expanded = tf.tile(Wa_expanded, [batch_size, 1, 1, 1]) # B x K x d x h1

    M = tf.matmul(embed_expanded, Wa_expanded) # B x K x n x h1

    P = []
    total_attn_outputs = []
    for k in range(num_aspect):
        weighted_outputs, attn_outputs = local_context_attention(M[:,k,:,:], filter_size=filter_size,
                        initializer=initializer, name='{}_{}'.format(k, name),
                        reuse=reuse)
        P.append(weighted_outputs) # B x h1
        total_attn_outputs.append(attn_outputs) # B x n

    outputs = tf.stack(P, axis=1) # B x K x h1
    total_attn_outputs = tf.stack(total_attn_outputs, axis=1) # B x K x n

    if(dropout is not None):
        outputs = tf.nn.dropout(outputs, dropout)

    return outputs, total_attn_outputs


def local_context_attention(inputs, filter_size=3, initializer=None,
                                reuse=None, name=''):

   # inputs: B x n x h1
    with tf.variable_scope('raw_aspect_embedding_att_{}'.format(name), reuse=reuse) as f:
        dim = inputs.get_shape().as_list()[2]
        filter_shape = [filter_size, dim, 1]

        W1 = tf.get_variable("weights", filter_shape,
                                initializer=initializer)
        conv =  tf.nn.conv1d(inputs, W1, stride=1,
                        padding="SAME", data_format="NHWC")
        # this should be bsz x seq_len x 1

        att = tf.nn.softmax(conv, axis=1) # B x n x 1
        weighted_inputs = tf.reduce_sum(inputs * att, 1) # B x n x h1 -> B x h1     
        att = tf.squeeze(att, axis=2)
        return weighted_inputs, att

