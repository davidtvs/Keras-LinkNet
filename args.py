from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them."""
    parser = ArgumentParser()

    # Execution mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',
        help=(
            "train: performs training and validation; test: tests the model "
            "found in \"--checkpoint-dir\" with name \"--name\" on \"--dataset\"; "
            "full: combines train and test modes. Default: train"
        )
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        help=(
            "The model found in \"--checkpoint-dir/--name/\" and filename "
            "\"--name.h5\" is loaded."
        )
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=0,
        help="Epoch at which to start training. Default: 0"
    )
    parser.add_argument(
        "--no-pretrained-encoder",
        dest='pretrained_encoder',
        action='store_false',
        help=(
            "Pretrained encoder weights are not loaded."
        )
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="./checkpoints/linknet_encoder_weights.h5",
        help=(
            "HDF5 file where the weights are stored. This setting is ignored "
            "if \"--no-pretrained-encoder\" is set. Default: "
            "/checkpoints/linknet_encoder_weights.h5"
        )
    )

    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="The batch size. Default: 10"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300"
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4"
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.1"
    )
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=100,
        help=(
            "The number of epochs before adjusting the learning rate. "
            "Default: 100"
        )
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes'],
        default='camvid',
        help="Dataset to use. Default: camvid"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/CamVid",
        help=(
            "Path to the root directory of the selected dataset. "
            "Default: data/CamVid"
        )
    )

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. Default: 4"
    )
    parser.add_argument(
        "--verbose",
        choices=[0, 1, 2],
        default=1,
        help=(
            "Verbosity mode: 0 - silent, 1 - progress bar, 2 - one line per "
            "epoch. Default: 1"
        )
    )

    # Storage settings
    parser.add_argument(
        "--name",
        type=str,
        default='LinkNet',
        help="Name given to the model when saving. Default: LinkNet"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default='checkpoints',
        help="The directory where models are saved. Default: checkpoints"
    )

    return parser.parse_args()
