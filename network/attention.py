import sys
import tensorflow as tf
sys.path.append('tools')
from hp import HP

class Attention(tf.keras.layers.Layer):
    def __init__(self, num_hidden_units, name="attention"):
        super().__init__(name=name)
        self.num_hidden_units = num_hidden_units
        
    def call(self, K, V, Q, training=False):
        # محاسبه توجه
        A = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(tf.cast(self.num_hidden_units, tf.float32))
        A = tf.nn.softmax(A)
        
        # محاسبه خروجی
        R = tf.matmul(A, V)
        
        # ترکیب با Q
        R = tf.concat((R, Q), -1)
        
        return R, A

def attention(K, V, Q, num_hidden_units, mononotic_attention=False, prev_max_attentions=None):
    """
    تابع سازگاری برای حفظ رابط قدیمی
    """
    attn = Attention(num_hidden_units=num_hidden_units)
    return attn(K, V, Q, training=True)
    