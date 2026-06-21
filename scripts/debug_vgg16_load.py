import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Attempting to load VGG16...")
model = tf.keras.applications.VGG16(weights='imagenet')
print("Model loaded successfully.")
