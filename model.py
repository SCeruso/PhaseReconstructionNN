import tensorflow as tf

def conv2d(inputs, filters, kernel_size, name, activation=None):
    convolved_t = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=1,
        padding='SAME', use_bias=False, name=name,
        kernel_initializer=tf.variance_scaling_initializer(), reuse=tf.AUTO_REUSE, activation=None)
    return convolved_t

def model(sx, sy, width=[32, 64, 64, 1024]):
    input_image = tf.transpose(tf.stack([sx, sy]), [1, 2, 3, 0])

    h_conv1 = conv2d(input_image, width[0], 7, name='conv1')
    h_pool1 = tf.layers.average_pooling2d(h_conv1, 2, 2)

    h_conv2 = conv2d(h_pool1, width[1], 7, name='conv2')
            
    h_pool2 = tf.layers.average_pooling2d(h_conv2, 2, 2)
            
    h_conv3 = conv2d(h_pool2, width[2], 7, name='conv3')

    h_pool3 = tf.layers.average_pooling2d(h_conv3, 2, 2)

    h_fc1 = conv2d(h_pool3, width[3], 1, name='conv4')

    h_resize1 = tf.cast(tf.image.resize_images(h_fc1, h_conv3.shape[1:3]), tf.float32)
    h_sharp = tf.concat([h_resize1, h_conv3], -1)
    h_res_in2 = conv2d(h_sharp, width[2], 7, name='conv5')
            
    h_resize2 = tf.cast(tf.image.resize_images(h_res_in2, h_conv2.shape[1:3]), tf.float32)
    h_sharp = tf.concat([h_resize2, h_conv2], -1)
    h_res_in3 = conv2d(h_sharp, width[1], 7, name='conv6')
            
    h_resize3 = tf.cast(tf.image.resize_images(h_res_in3, h_conv1.shape[1:3]), tf.float32)
    h_sharp = tf.concat([h_resize3, h_conv1], -1)
    res = conv2d(h_sharp, width[0], 7, name='conv7')

    res = conv2d(res, 1, 1, name='final')

    return res