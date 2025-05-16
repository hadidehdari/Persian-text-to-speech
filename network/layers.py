import tensorflow as tf

#https://www.w3cschool.cn/doc_tensorflow_python/tensorflow_python-tf-layers-conv1d.html
'''He initialization is implemented in variance_scaling_initializer(), but it didnt work for me (loss keeps blowing up) so i used Xavier initialization.i found the following questions useful:

https://stackoverflow.com/questions/43284047/what-is-the-default-kernel-initializer-in-tf-layers-conv2d-and-tf-layers-dense
https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-whatare'''

class Conv1DLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, dilation_rate, activation, dropout_rate=0.0, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        self.conv = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=activation,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )
        
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        # اعمال padding برای حالت causal
        if self.padding.lower() == "causal":
            pad_len = (self.kernel_size - 1) * self.dilation_rate
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"
        else:
            padding = self.padding
            
        x = self.conv(inputs)
        
        if self.dropout_rate > 0 and training:
            x = self.layer_norm(x)
            x = self.dropout(x)
            
        return x

# it's  actually simillar to gated convolution
#Highway(X; L) = σ(H1) * H2 + (1 − σ(H1))  X, where H1, H2 are properly-sized two matrices,
#output by a layer L as [H1, H2] = L(X). 
#The operator  is the element-wise multiplication, and σ is the element-wise sigmoid function.

class HighwayLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, dilation_rate, activation, dropout_rate=0.0, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        self.conv = Conv1DLayer(
            filters=2*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        _input = inputs
        
        # اگر filters مشخص نشده باشد، از اندازه ورودی استفاده می‌کنیم
        if self.filters is None:
            self.filters = inputs.get_shape().as_list()[-1]
            
        x = self.conv(inputs, training=training)
        
        # تقسیم به H1 و H2
        H1, H2 = tf.split(x, 2, axis=-1)
        
        if self.dropout_rate > 0 and training:
            H1 = self.layer_norm(H1)
            H2 = self.layer_norm(H2)
            
        # محاسبه خروجی highway
        gate = tf.nn.sigmoid(H1)
        output = gate * H2 + (1. - gate) * _input
        
        if self.dropout_rate > 0 and training:
            output = self.dropout(output)
            
        return output

#fancy deconvolution
class DeconvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, dropout_rate=0.0, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(1, kernel_size),
            strides=(1, strides),
            padding=padding,
            activation=activation,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )
        
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        # اگر filters مشخص نشده باشد، از اندازه ورودی استفاده می‌کنیم
        if self.filters is None:
            self.filters = inputs.get_shape().as_list()[-1]
            
        # اضافه کردن بعد برای کانولوشن 2D
        x = tf.expand_dims(inputs, 1)
        
        # اعمال deconvolution
        x = self.deconv(x)
        
        # حذف بعد اضافی
        x = tf.squeeze(x, 1)
        
        if self.dropout_rate > 0 and training:
            x = self.layer_norm(x)
            x = self.dropout(x)
            
        return x

# توابع سازگاری برای حفظ رابط قدیمی
def conv1d(input_tensor, filters, kernel_size, strides, padding, dilation_rate, activation, dropout_rate):
    layer = Conv1DLayer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        dropout_rate=dropout_rate
    )
    return layer(input_tensor, training=True)

def highwaynet(input_tensor, filters, kernel_size, strides, padding, dilation_rate, activation, dropout_rate, scope_name):
    layer = HighwayLayer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        dropout_rate=dropout_rate,
        name=scope_name
    )
    return layer(input_tensor, training=True)

def deconv(input_tensor, filters, kernel_size, strides, padding, activation, dropout_rate):
    layer = DeconvLayer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        dropout_rate=dropout_rate
    )
    return layer(input_tensor, training=True)