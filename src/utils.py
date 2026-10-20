import joblib


def save_object(filepath, obj):

    joblib.dump(
        obj,
        filepath
    )


def load_object(filepath):

    return joblib.load(
        filepath
    )