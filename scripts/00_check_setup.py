import ee, numpy, kagglehub, tensorflow as tf
print("All imports OK")
print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))