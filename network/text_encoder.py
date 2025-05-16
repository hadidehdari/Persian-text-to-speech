import tensorflow as tf
from layers import Conv1DLayer, HighwayLayer

class TextEncoder(tf.keras.layers.Layer):
    def __init__(self, num_hidden_layers, dropout_rate=0.5, name="text_encoder"):
        super().__init__(name=name)
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        
        # تعریف لایه‌های کانولوشن
        self.conv1 = Conv1DLayer(
            filters=num_hidden_layers*2,
            kernel_size=1,
            strides=1,
            padding="SAME",
            dilation_rate=1,
            activation=tf.nn.relu,
            dropout_rate=dropout_rate,
            name="conv1"
        )
        
        self.conv2 = Conv1DLayer(
            filters=num_hidden_layers*2,
            kernel_size=1,
            strides=1,
            padding="SAME",
            dilation_rate=1,
            activation=None,
            dropout_rate=dropout_rate,
            name="conv2"
        )
        
        # تعریف لایه‌های highway
        self.highway_layers = []
        for i in range(12):
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
        
        # لایه‌های نرمال‌سازی و dropout
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        # لایه کانولوشن اول
        x = self.conv1(inputs, training=training)
        
        # لایه کانولوشن دوم
        x = self.conv2(x, training=training)
        
        # لایه‌های highway
        for highway in self.highway_layers:
            x = highway(x, training=training)
        
        # تقسیم به K و V
        K, V = tf.split(x, 2, axis=-1)
        
        return K, V

class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, name="embedding"):
        super().__init__(name=name)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
            embeddings_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
            name="embedding_table"
        )
        
    def call(self, inputs):
        return self.embedding(inputs)

def textencoder(embeding_tensor, dropout_rate, num_hidden_layers):
    """
    تابع سازگاری برای حفظ رابط قدیمی
    """
    encoder = TextEncoder(num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    return encoder(embeding_tensor, training=True)

def embeding_layer(inputtextids, vocab_size, emdeding_size, scope_name, padding=False):
    """
    تابع سازگاری برای حفظ رابط قدیمی
    """
    embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_size=emdeding_size, name=scope_name)
    return embedding(inputtextids)
    