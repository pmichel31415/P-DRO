import tensorflow as tf


def tf_load_image(file_path, image_size=None):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if image_size is not None:
        img = tf.image.resize(img, [image_size[0], image_size[1]])
    return img.numpy()
