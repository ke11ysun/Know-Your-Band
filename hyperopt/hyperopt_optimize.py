
"""Auto-optimizing a neural network with Hyperopt (TPE algorithm)."""

# from neural_net import build_and_train, build_model
import build_and_train
from utils import print_json, save_json_result, load_best_hyperspace

from keras.utils import plot_model
import keras.backend as K
from hyperopt import hp, tpe, fmin, Trials

import pickle
import os
import traceback


space = {
    # 'optimizer_lr': hp.loguniform('optimizer_lr', -6, 0),
    'optimizer_lr': hp.uniform('optimizer_lr', -6, -3), #10**
    # 'sgd_lr': hp.loguniform('sgd_lr', -6, 0),
    'sgd_lr': hp.uniform('sgd_lr', -6, -1), #10**
    'optimizer_decay': hp.uniform('optimizer_decay', -6, 0),
    'sgd_decay': hp.uniform('sgd_decay', -6, 0),
    'num_rois': hp.choice('num_rois', [4,8,16,32,64]),
    'kernel_size': hp.quniform('kernel_size', 3,7,1)
    # 'anchor_box_scales': hp.choice('anchor_box_scales', [[128, 256, 512]]),
    # 'dropout': hp.choice('dropout', [None, hp.uniform('dropout_rate', 0.0, 0.6)])
}


def optimize_cnn(hype_space):
    """Build a convolutional neural network and train it."""
    try:
        model, model_name, result, _ = build_and_train(hype_space)

        # Save training results to disks with unique filenames
        save_json_result(model_name, result)

        K.clear_session()
        del model

        return result

    except Exception as err:
        try:
            K.clear_session()
        except:
            pass
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }

    print("\n\n")


def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        build_and_train.build_and_train,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")


if __name__ == "__main__":
    count = 0
    while count<5:
        count = count + 1
        print("Count: " + str(count))
        # Optimize a new model with the TPE Algorithm:
        print("OPTIMIZING NEW MODEL:")
        try:
            run_a_trial()
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)

