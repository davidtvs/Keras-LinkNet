from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, \
    Activation, Add, Input, Softmax
from keras.models import Model
from keras.backend import int_shape, is_keras_tensor
from .conv2d_transpose import Conv2DTranspose


class LinkNet():
    """LinkNet architecture.

    The model follows the architecture presented in: https://arxiv.org/abs/1707.03718

    Args:
        num_classes (int): the number of classes to segment.
        input_tensor (tensor, optional): Keras tensor
            (i.e. output of `layers.Input()`) to use as image input for
            the model. Default: None.
        input_shape (tuple, optional): Shape tuple of the model input.
            Default: None.
        initial_block_filters (int, optional): The number of filters after
            the initial block (see the paper for details on the initial
            block). Default: None.
        bias (bool, optional): If ``True``, adds a learnable bias.
            Default: ``False``.

    """

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
        self.output_shape = input_shape[:-1] + (num_classes, )

        # Create a Keras tensor from the input_shape/input_tensor
        if input_tensor is None:
            self.input = Input(shape=input_shape)
        elif is_keras_tensor(input_tensor):
            self.input = input_tensor
        else:
            # input_tensor is a tensor but not one from Keras
            self.input = Input(tensor=input_tensor, shape=input_shape)

    def get_model(self):
        """Initializes a LinkNet model.

        Returns:
            A Keras model instance.

        """
        # Initial block
        initial_block1 = Conv2D(
            self.initial_block_filters,
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=self.bias
        )(self.input)
        initial_block1 = BatchNormalization()(initial_block1)
        initial_block1 = Activation('relu')(initial_block1)
        initial_block2 = MaxPooling2D(pool_size=2)(initial_block1)

        # Encoder blocks
        encoder1 = self._encoder_block(
            initial_block2, self.initial_block_filters * 2, bias=self.bias
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
            encoder4,
            self.initial_block_filters * 8,
            output_shape=int_shape(encoder3)[1:],
            bias=self.bias
        )
        decoder = Add()([encoder3, decoder])
        decoder = self._decoder_block(
            decoder,
            self.initial_block_filters * 4,
            output_shape=int_shape(encoder2)[1:],
            bias=self.bias
        )
        decoder = Add()([encoder2, decoder])
        decoder = self._decoder_block(
            decoder,
            self.initial_block_filters * 2,
            output_shape=int_shape(encoder1)[1:],
            bias=self.bias
        )
        decoder = Add()([encoder1, decoder])
        decoder = self._decoder_block(
            decoder,
            self.initial_block_filters,
            output_shape=int_shape(initial_block2)[1:],
            bias=self.bias
        )

        # Final block
        # Build the output shape of the next layer - same width and height
        # as initial_block1
        shape = (
            int_shape(initial_block1)[1],
            int_shape(initial_block1)[2],
            self.initial_block_filters // 2,
        )
        decoder = Conv2DTranspose(
            self.initial_block_filters // 2,
            kernel_size=3,
            strides=2,
            padding='same',
            output_shape=shape,
            use_bias=self.bias
        )(decoder)
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
        decoder = Conv2DTranspose(
            self.num_classes,
            kernel_size=2,
            strides=2,
            padding='valid',
            output_shape=self.output_shape,
            use_bias=self.bias
        )(decoder)

        prediction = Softmax()(decoder)

        return Model(inputs=self.input, outputs=prediction)

    def _encoder_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        padding='same',
        bias=False,
    ):
        """Creates an encoder block.

        The encoder block is a combination of two basic encoder blocks
        (see ``_encoder_basic_block``). The first with stride 2 and the
        the second with stride 1.

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.

        Returns:
            The output tensor of the block.

        """
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
        """Creates a basic encoder block.

        Main brach architecture:
        1. Conv2D
        2. BatchNormalization
        3. ReLU
        Residual branch architecture:
        1. Conv2D, if `strides` is greater than 1
        The output of the main and residual branches are summed.

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            strides (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the strides along the height and width
                of the 2D input. In case it's a single integer, it's value
                is used for all spatial dimensions. Default: 1.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.

        Returns:
            The output tensor of the block.

        """
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
        strides=2,
        projection_ratio=4,
        padding='same',
        output_shape=None,
        bias=False
    ):
        """Creates a decoder block.

        Decoder block architecture:
        1. Conv2D
        2. BatchNormalization
        3. ReLU
        4. Conv2DTranspose
        5. BatchNormalization
        6. ReLU
        7. Conv2D
        8. BatchNormalization
        9. ReLU

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            strides (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the strides along the height and width
                of the 2D input. In case it's a single integer, it's value
                is used for all spatial dimensions. Default: 1.
            projection_ratio (int, optional): A scale factor applied to
                the number of input channels. The output of the first
                convolution will have ``input_channels // projection_ratio``.
                The goal is to decrease the number of parameters in the
                transposed convolution layer. Default: 4.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            output_shape: A tuple of integers specifying the shape of the output
                without the batch size. Default: None.
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.

        Returns:
            The output tensor of the block.

        """
        internal_filters = int_shape(input)[-1] // projection_ratio
        x = Conv2D(
            internal_filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=bias
        )(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # The shape of the following trasposed convolution is the output
        # shape of the block with 'internal_filters' channels
        shape = output_shape[:-1] + (internal_filters, )
        x = Conv2DTranspose(
            internal_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_shape=shape,
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
