from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0.8


# dataset functionality
def get_all_test_data(im_list, la_list):
    images = []
    labels = []
    index = 0
    for im_filename, la_filename in zip(im_list, la_list):
        im = Image.open(im_filename).resize((256,256))
        im = np.array(im, np.float32)
        im = im[np.newaxis]
        la = Image.open(la_filename).resize((256,256))
        la = np.array(la)
        la = la[np.newaxis]
        la = la[..., np.newaxis]
        images.append(im)
        labels.append(la)
    return images, labels


def get_filename_list(path):
    fd = open(path)
    image_filenames = []
    label_filenames = []
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])
    return image_filenames, label_filenames


def Dataset_reader(filename_queue,image_height,image_width,):
    image_filename = filename_queue[0]
    label_filename = filename_queue[1]

    imageValue = tf.read_file(image_filename)
    labelValue = tf.read_file(label_filename)

    image_bytes = tf.image.decode_png(imageValue)
    label_bytes = tf.image.decode_png(labelValue, dtype=tf.uint16)

    image = tf.reshape(image_bytes, (image_height, image_width,3))
    label = tf.reshape(label_bytes, (image_height, image_width, 1))
    # image = tf.image.resize_images(image, (256,256))
    # label = tf.image.resize_images(label, (256,256))
    image = tf.image.random_brightness(image,max_delta=63,seed=1)
    image = tf.image.random_contrast(image,lower=0.2,upper=1.8,seed=1)
    image = tf.image.random_flip_left_right(image,seed=1)
    image = tf.image.random_flip_up_down(image,seed=1)
    label = tf.image.random_flip_left_right(label,seed=1)
    label = tf.image.random_flip_up_down(label,seed=1)

    return image, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 3-D Tensor of [height, width, 1] type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 1
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # tf.image_summary('images', images)

    return images, label_batch


def DatasetInputs(image_filenames, label_filenames, batch_size,image_height,image_width):
    images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

    image, label = Dataset_reader(filename_queue,image_height,image_width)
    reshaped_image = tf.cast(image, tf.float32)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CamVid images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(reshaped_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def writeImage(image, filename):
    """ store label data to colored image """
    Sky = [128, 128, 128]
    Crack = [128, 0, 0]
    Unlabelled = [0, 0, 0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array(
        [Sky, Crack, Unlabelled])
    for l in range(0, 2):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

