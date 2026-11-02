import os, kagglehub
# this file is (Kaggle download + TFRecord parsing)

path = kagglehub.dataset_download("fantineh/next-day-wildfire-spread")
print("Downloaded to:", path)
for f in sorted(os.listdir(path)):
    print(f)



import glob
import tensorflow as tf

INPUTS = ["elevation", "th", "vs", "tmmn", "tmmx", "sph",
          "pr", "pdsi", "NDVI", "population", "erc", "PrevFireMask"]
OUTPUT = "FireMask"
SIZE = 64
desc = {k: tf.io.FixedLenFeature([SIZE, SIZE], tf.float32) for k in INPUTS + [OUTPUT]}

def parse(proto):
    ex = tf.io.parse_single_example(proto, desc)
    x = tf.stack([ex[k] for k in INPUTS], axis=-1)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    y = ex[OUTPUT][..., None]
    return x, y

files = sorted(glob.glob(os.path.join(path, "*train*")))
print("Train files found:", len(files))

ds = tf.data.TFRecordDataset(files).map(parse).batch(32)
x, y = next(iter(ds))
print("Input shape:", x.shape)    # expect (32, 64, 64, 12)
print("Label shape:", y.shape)    # expect (32, 64, 64, 1)
print("Label values:", sorted(set(y.numpy().flatten().tolist())))  # expect -1, 0, 1