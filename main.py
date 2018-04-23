from models.linknet import LinkNet
from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
from metrics.miou import MeanIoU

if __name__ == '__main__':
    args = get_arguments()

    # Import the requested dataset
    if args.dataset.lower() == 'camvid':
        from data import CamVidGenerator as DataGenerator
    else:
        # Should never happen...but just in case it does
        raise RuntimeError(
            "\"{0}\" is not a supported dataset.".format(args.dataset)
        )

    # Initialize the dataset generator
    train_generator = DataGenerator(
        args.dataset_dir, batch_size=args.batch_size
    )

    # Some information about the dataset
    image_batch, label_batch = train_generator[0]
    num_classes = label_batch[0].shape[-1]
    print("Training dataset size: {}".format(len(train_generator)))
    print("Image size {}".format(image_batch.shape))
    print("Label size {}".format(label_batch.shape))
    print("Number of classes (including unlabelled) {}".format(num_classes))

    # Compute class weights if needed
    print("Weighing technique: {}".format(args.weighing))
    if (args.weighing is not None):
        print("Computing class weights...")
        print("(this can take a while depending on the dataset size)")
        if args.weighing.lower() == 'enet':
            class_weights = enet_weighing(train_generator, num_classes)
        elif args.weighing.lower() == 'mfb':
            class_weights = median_freq_balancing(train_generator, num_classes)
        else:
            class_weights = None
    else:
        class_weights = None

    # Set the unlabelled class weight to 0 if requested
    if class_weights is not None:
        # Handle unlabelled class
        if args.ignore_unlabelled:
            if args.dataset.lower() == 'camvid':
                class_weights[-1] = 0

    print("Class weights: {}".format(class_weights))

    # Create the model
    model = LinkNet(num_classes, input_shape=image_batch[0].shape).get_model()
    print(model.summary())

    # Compile the model
    # Optimizer: Adam
    optim = Adam(args.learning_rate)

    # Initialize mIoU metric
    miou_metric = MeanIoU(num_classes)

    # Loss: Categorical crossentropy loss
    model.compile(
        optimizer=optim,
        loss='categorical_crossentropy',
        metrics=[metrics.categorical_accuracy, miou_metric.mean_iou]
    )

    # Set up learining rate scheduler
    def _lr_decay(epoch, lr):
        return args.lr_decay**(epoch // args.lr_decay_epochs) * args.learning_rate
    lr_scheduler = LearningRateScheduler(_lr_decay)

    # Train the model
    model.fit_generator(
        train_generator,
        class_weight=class_weights,
        epochs=args.epochs,
        callbacks=[lr_scheduler],
        workers=args.workers,
        verbose=args.verbose
    )
