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
            "found in \"--save_dir\" with name \"--name\" on \"--dataset\"; "
            "full: combines train and test modes. Default: train"
        )
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help=(
            "If set to true, the model found in \"--save_dir\" with name "
            "\"--name\" is loaded and training is resumed from the epoch it "
            "was saved in. Default: False"
        )
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=0,
        help="Epoch at which to start training. Default: 0"
    )
    parser.add_argument(
        "--pretrained_encoder",
        type=bool,
        default=True,
        help="Whether to load pretrained encoder weights or not. Default: True"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="./checkpoints/linknet_encoder_weights.h5",
        help=(
            "HDF5 file where the weights are stored. This setting is ignored "
            "if \"--pretrained_encoder\" is set to False. Default: "
            "/checkpoints/linknet_encoder_weights.h5"
        )
    )

    # Hyperparameters
    parser.add_argument(
        "--batch_size",
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
        "--learning_rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4"
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.1"
    )
    parser.add_argument(
        "--lr_decay_epochs",
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
        "--dataset_dir",
        type=str,
        default="data/CamVid",
        help=(
            "Path to the root directory of the selected dataset. "
            "Default: data/CamVid"
        )
    )
    parser.add_argument(
        "--weighing",
        choices=['enet', 'mfb', 'none'],
        default='ENet',
        help=(
            "The class weighing technique to apply to the dataset. "
            "Default: enet"
        )
    )
    parser.add_argument(
        "--ignore_unlabelled",
        type=bool,
        default=True,
        help=(
            "If True, the unlabelled class weight is ignored (set to 0); "
            "otherwise, it's kept as computed. Default: True"
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
        "--checkpoint_dir",
        type=str,
        default='checkpoints',
        help="The directory where models are saved. Default: checkpoints"
    )

    return parser.parse_args()
