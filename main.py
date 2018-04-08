from data import CamVidGenerator as DataGenerator
from models.linknet import LinkNet

if __name__ == '__main__':
    train_generator = DataGenerator(
        "/home/davidtvs/Deep-Learning/Datasets/CamVid", batch_size=10
    )
    print(len(train_generator))
    image_batch, label_batch = train_generator[0]
    print(image_batch.shape)
    print(label_batch.shape)

    model = LinkNet(
        label_batch.shape[-1], input_shape=image_batch[0].shape
    ).get_model()
    print(model.summary())
