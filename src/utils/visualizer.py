import collections
import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch


class Visualizer:
    def __init__(self, keys):
        plt.ion()
        self.wins = {k: None for k in keys}

    def display(self, image, key):
        n_images = len(image) if isinstance(image, (list, tuple)) else 1

        if self.wins[key] is None:
            self.wins[key] = plt.subplots(ncols=n_images)

        fig, ax = self.wins[key]
        n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1

        assert n_images == n_axes

        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image))
        else:
            for i in range(n_images):
                ax[i].cla()
                ax[i].set_axis_off()
                ax[i].imshow(self.prepare_img(image[i]))

        plt.draw()
        self.mypause(0.001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

    @staticmethod
    def on():
        plt.ion()


if __name__ == '__main__':
    Visualizer.on()
