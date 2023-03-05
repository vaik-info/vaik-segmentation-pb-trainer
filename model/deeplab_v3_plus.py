import tensorflow as tf


def prepare(num_classes, image_size):
    model_input = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1. / 255.)(model_input)
    backbone = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_tensor=x
    )
    x = backbone.get_layer("block_7_add").output
    x = DilatedSpatialPyramidPooling(x, int(6 * image_size / 520), int(12 * image_size / 520),
                                     int(18 * image_size / 520))

    input_a = tf.keras.layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = backbone.get_layer("block_2_expand_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = tf.keras.layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation='sigmoid')(x)
    return tf.keras.Model(inputs=model_input, outputs=model_output)


def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        use_bias=False,
):
    x = TpuConv2DLayer(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='same',
        use_bias=use_bias,
        kernel_initializer='he_normal',
    )(block_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def DilatedSpatialPyramidPooling(dspp_input, dilation_rate_s, dilation_rate_m, dilation_rate_l):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_s = convolution_block(dspp_input, kernel_size=3, dilation_rate=dilation_rate_s)
    out_m = convolution_block(dspp_input, kernel_size=3, dilation_rate=dilation_rate_m)
    out_l = convolution_block(dspp_input, kernel_size=3, dilation_rate=dilation_rate_l)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_s, out_m, out_l])
    output = convolution_block(x, kernel_size=1)
    return output


class TpuConv2DLayer(tf.keras.layers.Conv2D):
    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding="valid",
            data_format=None,
            dilation_rate=(1, 1),
            groups=1,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=tf.keras.activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
            bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            kernel_constraint=tf.keras.constraints.get(kernel_constraint),
            bias_constraint=tf.keras.constraints.get(bias_constraint),
            **kwargs
        )

    def call(self, inputs):
        output = tf.nn.conv2d(inputs, self.kernel,
                              strides=self.strides,
                              padding='SAME',
                              data_format='NHWC',
                              dilations=self.dilation_rate)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias, data_format='NHWC')

        if self.activation is not None:
            output = self.activation(output)
        return output
