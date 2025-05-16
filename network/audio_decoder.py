import tensorflow as tf
from layers import Conv1DLayer, HighwayLayer

class AudioDecoder(tf.keras.layers.Layer):
    def __init__(self, num_hidden_layers, num_mels, dropout_rate=0.5, name="audio_decoder"):
        super().__init__(name=name)
        self.num_hidden_layers = num_hidden_layers
        self.num_mels = num_mels
        self.dropout_rate = dropout_rate
        
        # تعریف لایه کانولوشن اول
        self.conv1 = Conv1DLayer(
            filters=num_hidden_layers,
            kernel_size=1,
            strides=1,
            padding="CAUSAL",
            dilation_rate=1,
            activation=None,
            dropout_rate=dropout_rate,
            name="conv1"
        )
        
        # تعریف لایه‌های highway
        self.highway_layers = []
        for i in range(9):
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
        
        # لایه کانولوشن نهایی
        self.final_conv = Conv1DLayer(
            filters=num_mels,
            kernel_size=1,
            strides=1,
            padding="CAUSAL",
            dilation_rate=1,
            activation=None,
            dropout_rate=dropout_rate,
            name="final_conv"
        )
        
        # لایه‌های نرمال‌سازی و dropout
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        # لایه کانولوشن اول
        x = self.conv1(inputs, training=training)
        
        # لایه‌های highway
        for highway in self.highway_layers:
            x = highway(x, training=training)
        
        # لایه کانولوشن نهایی
        logits = self.final_conv(x, training=training)
        Y = tf.nn.sigmoid(logits)
        
        return logits, Y

def audiodecoder(input_tensor, dropout_rate, num_hidden_layers, num_mels):
    """
    تابع سازگاری برای حفظ رابط قدیمی
    """
    decoder = AudioDecoder(
        num_hidden_layers=num_hidden_layers,
        num_mels=num_mels,
        dropout_rate=dropout_rate
    )
    return decoder(input_tensor, training=True)
    