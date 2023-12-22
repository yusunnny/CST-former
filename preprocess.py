# Extracts the features, labels, and normalizes the development and evaluation split features.

from cls import cls_feature_class as cls_feature_class
import parameters as parameters
import sys


def main(argv):
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once.

    # use parameter set defined by user
    task_id = '3' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=False)

    # Extract features and normalize them
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features()

    # Extract labels
    dev_feat_cls.extract_all_labels()

    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        if '2022' in params['dataset_dir']:
            train_splits = [[1, 2, 3]]
        elif '2023' in params['dataset_dir']:
            train_splits = [[1,2,3]]
    dev_feat_cls.rotate_features(params,train_splits)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)