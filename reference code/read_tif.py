import tensorflow_datasets as tfds
tfds.load("clic", batch_size=64, download=True)