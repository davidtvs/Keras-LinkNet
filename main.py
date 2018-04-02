from data import CamVid11Generator as DataGenerator

if __name__ == '__main__':
    train_generator = DataGenerator(
        "/media/davidtvs/Storage/Datasets/CamVid/SegNet", batch_size=10)

    for idx, batch in enumerate(train_generator):
        print(">>>> Batch index: ", idx)
        image_batch, label_batch = batch
        print(image_batch)
        print(label_batch)
        print()
        print(image_batch.shape)
        print(label_batch.shape)
        print()
