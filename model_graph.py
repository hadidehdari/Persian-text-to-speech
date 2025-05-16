import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write

sys.path.append('tools')
sys.path.append('network')

from read_data import load_data_main, get_batch, get_batch_npz
from hp import HP
from text_encoder import textencoder, embeding_layer
from audio_encoder import audioencoder
from audio_decoder import audiodecoder
from attention import attention
from SuperResolution import super_resolution
from read_data import load_data_synthesize
from wavprepro import spectrogram2wav

class TextToSpeechModel(tf.keras.Model):
    def __init__(self, HP, mode='demo'):
        super().__init__()
        self.HP = HP
        self.mode = mode
        
        # تعریف لایه‌های مدل
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=len(self.HP.persianvocab),
            output_dim=self.HP.embeding_num_units,
            name="embedding"
        )
        
        # تعریف بهینه‌ساز
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.HP.init_learinig_rate,
            beta_1=0.6,
            beta_2=0.95
        )
        
        # تعریف متغیر global_step
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    @tf.function
    def call(self, inputs, training=False):
        if self.mode == 'training_text2sp':
            texts, mels = inputs
            # پیش‌پردازش ورودی‌ها
            embedded_text = self.embedding_layer(texts)
            shifted_mels = tf.pad(mels[:,:-1,:], [[0,0], [1,0], [0,0]])
            
            # کدگذاری متن
            K, V = textencoder(embedded_text, dropout_rate=self.HP.dropout_rate, 
                             num_hidden_layers=self.HP.d)
            
            # کدگذاری صوت
            Q = audioencoder(shifted_mels, dropout_rate=self.HP.dropout_rate,
                           num_hidden_layers=self.HP.d)
            
            # توجه
            R, A = attention(K, V, Q, self.HP.d)
            
            # دیکودر صوت
            logits, Y = audiodecoder(R, dropout_rate=self.HP.dropout_rate,
                                   num_hidden_layers=self.HP.d,
                                   num_mels=self.HP.n_mels)
            
            return logits, Y, A
            
        elif self.mode == 'training_superresolution':
            mels, mags = inputs
            # شبکه سوپر رزولوشن
            logits, Z = super_resolution(mels, dropout_rate=self.HP.dropout_rate,
                                       num_hidden_layers=self.HP.c,
                                       n_fft=self.HP.n_fft)
            return logits, Z
            
        elif self.mode == 'demo':
            text_ids, prev_mels = inputs
            # پیش‌پردازش ورودی‌ها
            embedded_text = self.embedding_layer(text_ids)
            
            # کدگذاری متن
            K, V = textencoder(embedded_text, dropout_rate=self.HP.dropout_rate,
                             num_hidden_layers=self.HP.d)
            
            # کدگذاری صوت
            Q = audioencoder(prev_mels, dropout_rate=self.HP.dropout_rate,
                           num_hidden_layers=self.HP.d)
            
            # توجه
            R, A = attention(K, V, Q, self.HP.d)
            
            # دیکودر صوت
            logits, Y = audiodecoder(R, dropout_rate=self.HP.dropout_rate,
                                   num_hidden_layers=self.HP.d,
                                   num_mels=self.HP.n_mels)
            
            # سوپر رزولوشن
            mag_logits, Z = super_resolution(Y, dropout_rate=self.HP.dropout_rate,
                                           num_hidden_layers=self.HP.c,
                                           n_fft=self.HP.n_fft)
            
            return Y, Z, A

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            if self.mode == 'training_text2sp':
                logits, Y, A = self(inputs, training=True)
                # محاسبه loss
                l1_loss = tf.reduce_mean(tf.abs(inputs[1] - Y))
                binary_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=inputs[1])
                )
                # محاسبه attention loss
                N, T = tf.cast(tf.shape(A)[1], tf.float32), tf.cast(tf.shape(A)[2], tf.float32)
                W = tf.fill(tf.shape(A), 0.0)
                W = W + tf.expand_dims(tf.range(N), 1)/N - tf.expand_dims(tf.range(T), 0)/T
                att_W = 1.0 - tf.exp(-tf.square(W)/(2*0.2)**2)
                att_loss = tf.reduce_mean(tf.multiply(A, att_W))
                total_loss = l1_loss + binary_loss + att_loss
                
            elif self.mode == 'training_superresolution':
                logits, Z = self(inputs, training=True)
                l1_loss = tf.reduce_mean(tf.abs(inputs[1] - Z))
                binary_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=inputs[1])
                )
                total_loss = l1_loss + binary_loss
        
        # محاسبه گرادیان‌ها
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # clip گرادیان‌ها
        clipped_gradients = [tf.clip_by_value(grad, -1., 1.) for grad in gradients]
        # اعمال گرادیان‌ها
        self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))
        self.global_step.assign_add(1)
        
        return {'loss': total_loss}

    def predict(self, lines):
        lines = [item.replace("آ","آ").replace("أ","ا").replace("ئ","ی").replace("ؤ","و") for item in lines]
        Input_Text = load_data_synthesize(lines, self.HP.Max_Number_Of_Chars)
        predicted_mel = np.zeros((Input_Text.shape[0], self.HP.Max_Number_Of_MelFrames, self.HP.n_mels))
        
        print('predicting ( . •́ _ʖ •̀ .) ...')
        for i in tqdm(range(1, self.HP.Max_Number_Of_MelFrames)):
            previous_slice = predicted_mel[:,:i,:]
            mel_out, _, _ = self([Input_Text, previous_slice], training=False)
            predicted_mel[:,i,:] = mel_out.numpy()[:,-1,:]
        
        _, predicted_mag, _ = self([Input_Text, predicted_mel], training=False)
        
        print('converting the generated spectrogram to audio ｖ(⌒ｏ⌒)ｖ♪ ...')
        for i, mag in enumerate(predicted_mag.numpy()):
            wav = spectrogram2wav(mag)
            write('generated_samples/' + "/{}.wav".format(i+1), self.HP.sr, wav.astype(np.float32))
        print('DONE!')
