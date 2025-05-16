import tensorflow as tf
from layers import Conv1DLayer, HighwayLayer

class AudioEncoder(tf.keras.layers.Layer):
    def __init__(self, num_hidden_layers, dropout_rate=0.5, name="audio_encoder"):
        super().__init__(name=name)
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        
        # تعریف لایه‌های کانولوشن
        self.conv1 = Conv1DLayer(
            filters=num_hidden_layers,
            kernel_size=1,
            strides=1,
            padding="CAUSAL",
            dilation_rate=1,
            activation=tf.nn.relu,
            dropout_rate=dropout_rate,
            name="conv1"
        )
        
        self.conv2 = Conv1DLayer(
            filters=num_hidden_layers,
            kernel_size=1,
            strides=1,
            padding="CAUSAL",
            dilation_rate=1,
            activation=tf.nn.relu,
            dropout_rate=dropout_rate,
            name="conv2"
        )
        
        self.conv3 = Conv1DLayer(
            filters=num_hidden_layers,
            kernel_size=1,
            strides=1,
            padding="CAUSAL",
            dilation_rate=1,
            activation=None,
            dropout_rate=dropout_rate,
            name="conv3"
        )
        
        # تعریف لایه‌های highway
        self.highway_layers = []
        for i in range(10):
            dilation_rate = 1 if i % 4 == 0 else (3 if i % 4 == 1 else (9 if i % 4 == 2 else 27))
            self.highway_layers.append(
                HighwayLayer(
                    filters=None,
                    kernel_size=3,
                    strides=1,
                    padding="CAUSAL",
                    dilation_rate=dilation_rate,
                    activation=None,
                    dropout_rate=dropout_rate,
                    name=f"highway_{i+1}"
                )
            )
        
        # لایه‌های نرمال‌سازی و dropout
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        # لایه‌های کانولوشن
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        
        # لایه‌های highway
        for highway in self.highway_layers:
            x = highway(x, training=training)
        
        return x

def audioencoder(input_tensor, dropout_rate, num_hidden_layers):
    """
    تابع سازگاری برای حفظ رابط قدیمی
    """
    encoder = AudioEncoder(num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    return encoder(input_tensor, training=True)
    