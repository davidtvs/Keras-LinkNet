from data import CamVidGenerator as DataGenerator

if __name__ == '__main__':
    train_generator = DataGenerator(
        "/home/davidtvs/Deep-Learning/Datasets/CamVid",
        batch_size=10,
    )


