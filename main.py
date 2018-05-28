import os
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler, TensorBoard, \
    ModelCheckpoint, EarlyStopping
from args import get_arguments
from models.linknet import LinkNet
from models.conv2d_transpose import Conv2DTranspose
from metrics.miou import MeanIoU
from data.utils import enet_weighing, median_freq_balancing
from callbacks import TensorBoardPrediction


def train(
    epochs,
    initial_epoch,
    train_generator,
    val_generator,
    class_weights,
    learning_rate,
    lr_decay,
    lr_decay_epochs,
    pretrained_encoder='True',
    weights_path='./checkpoints/linknet_encoder_weights.h5',
    checkpoint_model=None,
    verbose=1,
    workers=1,
    checkpoint_path='./checkpoints',
    tensorboard_logdir='./checkpoints'
):
    # Create the model
    image_batch, label_batch = train_generator[0]
    num_classes = label_batch[0].shape[-1]
    input_shape = image_batch[0].shape
    if checkpoint_model is None:
        model = LinkNet(num_classes, input_shape=input_shape)
        model = model.get_model(
            pretrained_encoder=pretrained_encoder, weights_path=weights_path
        )
    else:
        model = checkpoint_model

    print(model.summary())

    # Optimizer: Adam
    optim = RMSprop(learning_rate)

    # Initialize mIoU metric
    miou_metric = MeanIoU(num_classes)

    # Compile the model
    # Loss: Categorical crossentropy loss
    model.compile(
        optimizer=optim,
        loss='categorical_crossentropy',
        metrics=['accuracy', miou_metric.mean_iou]
    )

    # Set up learining rate scheduler
    def _lr_decay(epoch, lr):
        return lr_decay**(epoch // lr_decay_epochs) * learning_rate

    lr_scheduler = LearningRateScheduler(_lr_decay)

    # TensorBoard callback
    tensorboard = TensorBoard(
        log_dir=tensorboard_logdir,
        histogram_freq=0,
        write_graph=True,
        write_images=True
    )

    # Tensorboard callback that displays a random sample with respective
    # target and prediction
    tensorboard_viz = TensorBoardPrediction(
        val_generator,
        val_generator.get_class_rgb_encoding(),
        log_dir=tensorboard_logdir
    )

    # Checkpoint callback - save the best model
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_mean_iou',
        save_best_only=True,
        mode='max'
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_mean_iou', min_delta=0.01, patience=10, mode='max'
    )

    callbacks = [
        lr_scheduler, tensorboard, tensorboard_viz, checkpoint, early_stop
    ]

    # Train the model
    model.fit_generator(
        train_generator,
        class_weight=class_weights,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        workers=workers,
        verbose=verbose,
        use_multiprocessing=True,
        validation_data=val_generator
    )

    return model


def test(model, test_generator, workers, verbose):
    metrics = model.evaluate_generator(
        test_generator,
        workers=workers,
        use_multiprocessing=True,
        verbose=verbose,
    )

    print("--> Evaluation metrics")
    for idx, value in enumerate(metrics):
        print("{0}: {1}".format(model.metrics_names[idx], value))

    return model


def main():
    # Get command line arguments
    args = get_arguments()

    # Import the desired dataset generator
    if args.dataset.lower() == 'camvid':
        from data import CamVidGenerator as DataGenerator
    elif args.dataset.lower() == 'cityscapes':
        from data import CityscapesGenerator as DataGenerator
    else:
        # Should never happen...but just in case it does
        raise RuntimeError(
            "\"{0}\" is not a supported dataset.".format(args.dataset)
        )

    # Initialize training and validation dataloaders
    if args.mode.lower() in ('train', 'full'):
        train_generator = DataGenerator(
            args.dataset_dir,
            batch_size=args.batch_size,
            mode='train',
            ignore_unlabeled=args.ignore_unlabeled
        )
        val_generator = DataGenerator(
            args.dataset_dir,
            batch_size=args.batch_size,
            mode='val',
            ignore_unlabeled=args.ignore_unlabeled
        )

        # Some information about the dataset
        image_batch, label_batch = train_generator[0]
        num_classes = label_batch[0].shape[-1]
        print("--> Training batches: {}".format(len(train_generator)))
        print("--> Validation batches: {}".format(len(val_generator)))
        print("--> Image size: {}".format(image_batch.shape))
        print("--> Label size: {}".format(label_batch.shape))
        print("--> No. of classes: {}".format(num_classes))

        # Compute class weights if needed
        print("--> Weighing technique: {}".format(args.weighing))
        class_weights = None
        if (args.weighing is not None):
            print("--> Computing class weights...")
            print("--> (this can take a while depending on the dataset size)")
            if args.weighing.lower() == 'enet':
                class_weights = enet_weighing(train_generator, num_classes)
            elif args.weighing.lower() == 'mfb':
                class_weights = median_freq_balancing(
                    train_generator, num_classes
                )

        print("--> Class weights: {}".format(class_weights))

    # Initialize test dataloader
    if args.mode.lower() in ('test', 'full'):
        test_generator = DataGenerator(
            args.dataset_dir,
            batch_size=args.batch_size,
            mode='test',
            ignore_unlabeled=args.ignore_unlabeled
        )

        # Some information about the dataset
        image_batch, label_batch = test_generator[0]
        num_classes = label_batch[0].shape[-1]
        print("--> Testing batches: {}".format(len(test_generator)))
        print("--> Image size: {}".format(image_batch.shape))
        print("--> Label size: {}".format(label_batch.shape))
        print("--> No. of classes: {}".format(num_classes))

    checkpoint_path = os.path.join(
        args.checkpoint_dir, args.name, args.name + '.h5'
    )
    print("--> Checkpoint path: {}".format(checkpoint_path))

    model = None

    if args.mode.lower() in ('train', 'full'):
        if args.resume:
            print("--> Resuming model: {}".format(checkpoint_path))
            model = load_model(
                checkpoint_path,
                custom_objects={
                    'Conv2DTranspose': Conv2DTranspose,
                    'mean_iou': MeanIoU(num_classes).mean_iou
                }
            )
        tensorboard_logdir = os.path.join(args.checkpoint_dir, args.name)
        model = train(
            args.epochs,
            args.initial_epoch,
            train_generator,
            val_generator,
            class_weights,
            args.learning_rate,
            args.lr_decay,
            args.lr_decay_epochs,
            pretrained_encoder=args.pretrained_encoder,
            weights_path=args.weights_path,
            checkpoint_model=model,
            verbose=args.verbose,
            workers=args.workers,
            checkpoint_path=checkpoint_path,
            tensorboard_logdir=tensorboard_logdir,
        )

    if args.mode.lower() in ('test', 'full'):
        print("--> Loading model: {}".format(checkpoint_path))
        model = load_model(
            checkpoint_path,
            custom_objects={
                'Conv2DTranspose': Conv2DTranspose,
                'mean_iou': MeanIoU(num_classes).mean_iou
            }
        )
        model = test(model, test_generator, args.workers, args.verbose)


if __name__ == '__main__':
    main()
