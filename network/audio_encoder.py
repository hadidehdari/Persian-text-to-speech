import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from layers import conv1d, highwaynet

def audioencoder(input_tensor, dropout_rate, num_hidden_layers):
    L1 = conv1d(input_tensor=input_tensor,
                filters=num_hidden_layers,
                kernel_size=1,
                strides=1,
                padding="CAUSAL",
                dilation_rate=1,
                activation=tf.nn.relu,
                dropout_rate=dropout_rate)
    
    L2 = conv1d(input_tensor=L1,
                filters=None,
                kernel_size=1,
                strides=1,
                padding="CAUSAL",
                dilation_rate=1,
                activation=tf.nn.relu,
                dropout_rate=dropout_rate)
    
    L3 = conv1d(input_tensor=L2,
                filters=None,
                kernel_size=1,
                strides=1,
                padding="CAUSAL",
                dilation_rate=1,
                activation=None,
                dropout_rate=dropout_rate)

    L4 = highwaynet(input_tensor=L3,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=1,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audioencoder_highwaynet_Block1")
                    
    L5 = highwaynet(input_tensor=L4,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=3,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audioencoder_highwaynet_Block2")
                    
    L6 = highwaynet(input_tensor=L5,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=9,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audioencoder_highwaynet_Block3")
                    
    L7 = highwaynet(input_tensor=L6,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=27,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audioencoder_highwaynet_Block4")
    
    L8 = highwaynet(input_tensor=L7,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=1,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audioencoder_highwaynet_Block5")
                    
    L9 = highwaynet(input_tensor=L8,
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=3,
                    activation=None,
                    dropout_rate=dropout_rate,
                    scope_name="audioencoder_highwaynet_Block6")
                    
    L10 = highwaynet(input_tensor=L9,
                     filters=None,
                     kernel_size=3,
                     strides=1,
                     padding="CAUSAL",
                     dilation_rate=9,
                     activation=None,
                     dropout_rate=dropout_rate,
                     scope_name="audioencoder_highwaynet_Block7")
                     
    L11 = highwaynet(input_tensor=L10,
                     filters=None,
                     kernel_size=3,
                     strides=1,
                     padding="CAUSAL",
                     dilation_rate=27,
                     activation=None,
                     dropout_rate=dropout_rate,
                     scope_name="audioencoder_highwaynet_Block8")
    
    L12 = highwaynet(input_tensor=L11,
                     filters=None,
                     kernel_size=3,
                     strides=1,
                     padding="CAUSAL",
                     dilation_rate=3,
                     activation=None,
                     dropout_rate=dropout_rate,
                     scope_name="audioencoder_highwaynet_Block9")
                     
    L13 = highwaynet(input_tensor=L12,
                     filters=None,
                     kernel_size=3,
                     strides=1,
                     padding="CAUSAL",
                     dilation_rate=3,
                     activation=None,
                     dropout_rate=dropout_rate,
                     scope_name="audioencoder_highwaynet_Block10")

    return L13
    