from segmentation import learn_seg, segment, MODEL_PATH
from tuning import tune_model


if __name__ == '__main__':
    learn_seg()
    segment('src/models/seg_model_116', override=True)
    tune_model()

