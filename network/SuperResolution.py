import tensorflow as tf
from layers import Conv1DLayer, HighwayLayer, DeconvLayer

class SuperResolution(tf.keras.layers.Layer):
    def __init__(self, num_hidden_layers, n_fft, dropout_rate=0.5, name="super_resolution"):
        super().__init__(name=name)
        self.num_hidden_layers = num_hidden_layers
        self.n_fft = n_fft
        self.dropout_rate = dropout_rate
        
        # تعریف لایه کانولوشن اول
        self.conv1 = Conv1DLayer(
            filters=num_hidden_layers,
            kernel_size=1,
            strides=1,
            padding="SAME",
            dilation_rate=1,
            activation=None,
            dropout_rate=dropout_rate,
            name="conv1"
        )
        
        # تعریف لایه‌های highway
        self.highway_layers = []
        for i in range(8):
            dilation_rate = 1 if i % 4 == 0 else (3 if i % 4 == 1 else (9 if i % 4 == 2 else 27))
            self.highway_layers.append(
                HighwayLayer(
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="SAME",
                    dilation_rate=dilation_rate,
                    activation=None,
                    dropout_rate=dropout_rate,
                    name=f"highway_{i+1}"
                )
            )
        
        # تعریف لایه‌های deconv
        self.deconv_layers = []
        for i in range(2):
            self.deconv_layers.append(
                DeconvLayer(
                    filters=num_hidden_layers,
                    kernel_size=3,
                    strides=2,
                    padding="SAME",
                    activation=None,
                    dropout_rate=dropout_rate,
                    name=f"deconv_{i+1}"
                )
            )
        
        # لایه‌های کانولوشن نهایی
        self.final_convs = [
            Conv1DLayer(
                filters=2*num_hidden_layers,
                kernel_size=1,
                strides=1,
                padding="SAME",
                activation=None,
                dropout_rate=dropout_rate,
                name="final_conv1"
            ),
            Conv1DLayer(
                filters=1+n_fft//2,
                kernel_size=1,
                strides=1,
                padding="SAME",
                activation=None,
                dropout_rate=dropout_rate,
                name="final_conv2"
            ),
            Conv1DLayer(
                filters=num_hidden_layers,
                kernel_size=1,
                strides=1,
                padding="SAME",
                activation=tf.nn.relu,
                dropout_rate=dropout_rate,
                name="final_conv3"
            ),
            Conv1DLayer(
                filters=num_hidden_layers,
                kernel_size=1,
                strides=1,
                padding="SAME",
                activation=tf.nn.relu,
                dropout_rate=dropout_rate,
                name="final_conv4"
            ),
            Conv1DLayer(
                filters=num_hidden_layers,
                kernel_size=1,
                strides=1,
                padding="SAME",
                activation=None,
                dropout_rate=dropout_rate,
                name="final_conv5"
            )
        ]
        
        # لایه‌های نرمال‌سازی و dropout
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        # لایه کانولوشن اول
        x = self.conv1(inputs, training=training)
        
        # لایه‌های highway اول
        for i in range(2):
            x = self.highway_layers[i](x, training=training)
        
        # لایه deconv اول
        x = self.deconv_layers[0](x, training=training)
        
        # لایه‌های highway دوم
        for i in range(2, 4):
            x = self.highway_layers[i](x, training=training)
        
        # لایه deconv دوم
        x = self.deconv_layers[1](x, training=training)
        
        # لایه‌های highway سوم
        for i in range(4, 6):
            x = self.highway_layers[i](x, training=training)
        
        # لایه‌های کانولوشن نهایی
        for conv in self.final_convs:
            x = conv(x, training=training)
        
        Z = tf.nn.sigmoid(x)
        
        return x, Z

def super_resolution(input_tensor, dropout_rate, num_hidden_layers, n_fft):
    """
    تابع سازگاری برای حفظ رابط قدیمی
    """
    sr = SuperResolution(
        num_hidden_layers=num_hidden_layers,
        n_fft=n_fft,
        dropout_rate=dropout_rate
    )
    return sr(input_tensor, training=True)
    