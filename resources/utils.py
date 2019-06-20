import os.path
import numpy as np

import tensorflow as tf

NUM_CLASSES = 2

# This function takes image and resizes it to smaller format (150x150)
def image_resize(img_path):
    # Be very careful with resizing images like this and make sure to read the doc!
    # Otherwise, bad things can happen - https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
    img_decoded = tf.image.decode_jpeg(img_path, channels=3, name="decoded_images")
    resized_image = tf.image.resize_images(img_decoded, (100, 100), align_corners=True)
    resized_image_asint = tf.cast(resized_image, tf.int32)
    return resized_image_asint

# Execution plan is defined here.
# Since it uses lazy evaluation, the images will not be read after calling build_pipeline_plan()
# We need to use iterator defined here in tf context
def build_pipeline_plan(img_paths, batch_size):

    # We build a tensor of image paths and labels
    tr_data = tf.data.Dataset.from_tensor_slices(img_paths)
    # Apply resize to each image in the pipeline
    tr_data_imgs = tr_data.map(image_resize)
    # Gives us opportuinty to batch images into small groups
    tr_dataset = tr_data_imgs.batch(batch_size)
    # create TensorFlow Iterator object directly from input pipeline
    iterator = tr_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element

# Function to execute defined pipeline in Tensorflow session
def process_pipeline(next_element):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # get each element of the training dataset until the end is reached
        # in our case only one iteration since we read everything as 1 batch
        # can be multiple iterations if we decrease BATCH_SIZE to eg. 10
        images = []
        while True:
            try:
                elem = sess.run(next_element)
                images = elem[0]
            except tf.errors.OutOfRangeError:
                print("Finished reading the image")
                return images


def load_image(path, batch_size):
    files = np.asarray([path])
    p = tf.constant(files, name="train_imgs")
    next_element = build_pipeline_plan(p, batch_size=batch_size)
    imgs = process_pipeline(next_element)
    return imgs

ALLOWED_EXTENSIONS = set(['jpg','jpeg'])
def allowed_file(filename):
    """Only .jpg files allowed"""
    return True
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_classification(image,sess,graph,model):
    #mg_path = request.files["image"].read()
    class_labels = {0: 'Cat', 1: 'Dog'}
    img = image.read()
    with sess.as_default():
        with graph.as_default():
            x_test = load_image(img, 6000)
            score = model.predict(np.asarray([x_test]))
    result = score[0].tolist()
    print(result)
    if result[0] >0.5:
        print("cat")
        return class_labels[0]
    else:
        print("dog")
        return class_labels[1]



