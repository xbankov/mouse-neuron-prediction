import os
from pathlib import Path

from fastai.vision import ImageList, load_data, DataBunch

from load_data import load_single_trial, load_classes


def prepare_data():
    classes = load_classes()
    trial = load_single_trial(classes)
    trial.save_blended_images()
    return trial.get_labeling_df()


# TODO add logger ;)
# noinspection PyArgumentList
def main():
    labels_df = prepare_data()
    bs = 32
    path = os.getcwd()
    filename = 'preprocessed_databunch.pkl'

    if Path(path, filename).is_file():
        print("Loading preprocessed databunch")
        data = load_data(path, filename)

    else:
        print("Preprocessing images")
        data: DataBunch = (ImageList.from_df(labels_df, os.getcwd(), convert_mode='L')
                           .split_by_rand_pct(0.2, 123)
                           .label_from_df()
                           .transform(size=128)
                           .databunch(bs=bs)
                           # .normalize()
                           )

        data.save(filename)

    # random_model = RandomNetwork()
    # my_learner = Learner(data, random_model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

    # model = ResidualNetwork()
    # my_learner = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
    # my_learner.lr_find(end_lr=100)
    # my_learner.recorder.plot()

    # my_learner.fit_one_cycle(10, 1e-2)
    # my_learner.save('INS_10_epochs')


if __name__ == '__main__':
    main()
