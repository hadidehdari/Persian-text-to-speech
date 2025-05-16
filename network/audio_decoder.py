import tensorflow as tf
from layers import conv1d, highwaynet

def audiodecoder(input_tensor, dropout_rate, num_hidden_layers, num_mels):
    L1 = conv1d(input_tensor=input_tensor,
                filters=num_hidden_layers,
                kernel_size=1,
                strides=1,
                padding="CAUSAL",
                dilation_rate=1,
                activation=None,
                dropout_rate=dropout_rate)

    L2 = highwaynet(input_tensor=L1,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=1,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audiodecoder_highwaynet_Block1")
                    
    L3 = highwaynet(input_tensor=L2,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=3,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audiodecoder_highwaynet_Block2")
                    
    L4 = highwaynet(input_tensor=L3,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=9,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audiodecoder_highwaynet_Block3")
                    
    L5 = highwaynet(input_tensor=L4,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=27,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audiodecoder_highwaynet_Block4")

    L6 = highwaynet(input_tensor=L5,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=1,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audiodecoder_highwaynet_Block5")  
                    
    L7 = highwaynet(input_tensor=L6,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=1,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audiodecoder_highwaynet_Block6")
    
    L8 = highwaynet(input_tensor=L7,
                    filters=None,
                    kernel_size=1,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=1,
                    activation=tf.nn.relu,
                    dropout_rate=dropout_rate,
                    scope_name="audiodecoder_highwaynet_Block7")
                    
    L9 = highwaynet(input_tensor=L8,
                    filters=None,
                    kernel_size=1,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=1,
                    activation=tf.nn.relu,
                    dropout_rate=dropout_rate,
                    scope_name="audiodecoder_highwaynet_Block8")
                    
    L10 = highwaynet(input_tensor=L9,
                     filters=None,
                     kernel_size=1,
                     strides=1,
                     padding="CAUSAL",
                     dilation_rate=1,
                     activation=tf.nn.relu,
                     dropout_rate=dropout_rate,
                     scope_name="audiodecoder_highwaynet_Block9")

    logits = conv1d(input_tensor=L10,
                    filters=num_mels,
                    kernel_size=1,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=1,
                    activation=None,
                    dropout_rate=dropout_rate)
    
    Y = tf.nn.sigmoid(logits)

    return logits, Y
    