"""Readers: models that turn a line image into text. experiment.py picks one by config["model"].

Every reader is trained inside one fold, on the training part only, and predicts the held-out part. A reader takes
a `ctx` (see autoresearch/remote.py) for logging, a kernel-lifetime cache and a work folder.
"""
