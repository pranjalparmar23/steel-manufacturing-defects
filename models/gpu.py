#%%
import tensorflow as tf

# Set operations to run on GPU if available
with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    c = a + b

print("Result:", c)

# %%
