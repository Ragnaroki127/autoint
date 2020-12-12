import tensorflow as tf

data_path = './criteo_sampled_data.csv'

k = 16

d = 32

H = 2

n_layers = 3

epochs = 15

batch_size = 1024

gpu_devices = '4, 5, 6, 7' 

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

