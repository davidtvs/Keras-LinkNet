from keras.layers import Conv2D, MaxPooling2D, \
    BatchNormalization, Activation, Add, Input, Lambda
from keras.models import Model
from keras.backend import int_shape, is_keras_tensor
from .conv2d_transpose import Conv2DTranspose


class LinkNet():
    def __init__(
        self,
        num_classes,
        input_tensor=None,
        input_shape=None,
        initial_block_filters=64,
        bias=False
    ):
        self.num_classes = num_classes
        self.initial_block_filters = initial_block_filters
        self.bias = bias

        # Create a Keras tensor from the input_shape/input_tensor
        if input_tensor is None:
            self.input = Input(shape=input_shape)
        elif is_keras_tensor(input_tensor):
            self.input = input_tensor
        else:
            # input_tensor is a tensor but not one from Keras
            self.input = Input(tensor=input_tensor, shape=input_shape)

    def get_model(self):
        # Initial block
        initial_block = Conv2D(
            self.initial_block_filters,
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=self.bias
        )(self.input)
        initial_block = BatchNormalization()(initial_block)
        initial_block = Activation('relu')(initial_block)
        initial_block = MaxPooling2D(pool_size=2)(initial_block)

        # Encoder blocks
        encoder1 = self._encoder_block(
            initial_block, self.initial_block_filters * 2, bias=self.bias
        )
        encoder2 = self._encoder_block(
            encoder1, self.initial_block_filters * 4, bias=self.bias
        )
        encoder3 = self._encoder_block(
            encoder2, self.initial_block_filters * 8, bias=self.bias
        )
        encoder4 = self._encoder_block(
            encoder3, self.initial_block_filters * 16, bias=self.bias
        )

        # Decoder blocks
        decoder = self._decoder_block(
            encoder4, self.initial_block_filters * 8, output_padding=(1, 0)
        )
        decoder = Add()([encoder3, decoder])
        decoder = self._decoder_block(
            decoder, self.initial_block_filters * 4, output_padding=(0, 1)
        )
        decoder = Add()([encoder2, decoder])
        decoder = self._decoder_block(
            decoder, self.initial_block_filters * 2, output_padding=(0, 1)
        )
        decoder = Add()([encoder1, decoder])
        decoder = self._decoder_block(
            decoder, self.initial_block_filters, output_padding=1
        )

        # Final block
        decoder = Conv2DTranspose(
            self.initial_block_filters // 2,
            kernel_size=3,
            strides=2,
            padding='same',
            output_padding=1,
            use_bias=self.bias
        )(decoder)
        print("decoder5", int_shape(decoder))
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(
            self.initial_block_filters // 2,
            kernel_size=3,
            padding='same',
            use_bias=self.bias
        )(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        prediction = Conv2DTranspose(
            self.num_classes,
            kernel_size=2,
            strides=2,
            padding='valid',
            output_padding=0,
            use_bias=self.bias
        )(decoder)

        return Model(inputs=self.input, outputs=prediction)

    def _encoder_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        padding='same',
        bias=False,
    ):
        x = self._encoder_basic_block(
            input,
            out_filters,
            kernel_size=kernel_size,
            strides=2,
            padding=padding,
            bias=bias
        )

        x = self._encoder_basic_block(
            x,
            out_filters,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            bias=bias
        )

        return x

    def _encoder_basic_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        bias=False
    ):
        residual = input

        x = Conv2D(
            out_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=bias
        )(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if strides > 1:
            residual = Conv2D(
                out_filters,
                kernel_size=1,
                strides=strides,
                padding=padding,
                use_bias=bias
            )(residual)

        x = Add()([x, residual])

        return x

    def _decoder_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        upsample_strides=2,
        projection_ratio=4,
        padding='same',
        output_padding=0,
        bias=False
    ):
        internal_filters = int_shape(input)[-1] // 4
        x = Conv2D(
            internal_filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=bias
        )(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(
            internal_filters,
            kernel_size=kernel_size,
            strides=upsample_strides,
            padding=padding,
            output_padding=output_padding,
            use_bias=bias
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(
            out_filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=bias
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x
