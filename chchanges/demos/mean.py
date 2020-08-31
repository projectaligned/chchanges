import imageio as imageio
import numpy as np
from matplotlib import pyplot as plt

from chchanges.bayesian_online import ConstantHazard, StudentT, Detector


def detect_mean_shift():
    normal_signal = np.random.normal(loc=50, scale=10, size=1000)
    normal_signal[250:500] += 30
    normal_signal[500:750] -= 30
    lambda_ = 100
    delay = 150

    hazard = ConstantHazard(lambda_)
    posterior = StudentT(var=1, df=1., mean=50, plot=True)
    detector = Detector(hazard, posterior, delay, threshold=0.25)

    data_plotter_fig, data_plotter_ax = plt.subplots()
    data_plotter_ax.set_title('Data Stream')
    data_plotter_ax.set_xlabel('Datum index')
    data_plotter_ax.set_ylabel('Datum value')

    prob_plotter_fig, prob_plotter_ax = plt.subplots()
    prob_plotter_ax.set_title('Probability Stream')
    prob_plotter_ax.set_xlabel('Datum index')
    prob_plotter_ax.set_ylabel('Probability of changepoint')

    idxs_so_far = []
    for idx, datum in enumerate(normal_signal):
        idxs_so_far.append(idx)
        changepoint_detected = detector.update(datum)
        detector.posterior.update_plot(live=True)
        data_plotter_ax.errorbar(idx, datum, fmt='k.', alpha=0.3)
        if idx > delay:
            prob_plotter_ax.errorbar(idx, detector.growth_probs[delay], fmt='k.', alpha=0.3)
        if changepoint_detected:
            changepoint_idx = idxs_so_far[-delay]
            data_plotter_ax.axvline(changepoint_idx, alpha=0.5, color='r', linestyle='--')
        yield prob_plotter_fig, data_plotter_fig, detector.posterior.fig
    plt.show()


def make_gifs():
    prob_images = []
    data_images = []
    post_images = []
    for prob_fig, data_fig, post_fig in detect_mean_shift():
        prob_image = np.frombuffer(prob_fig.canvas.tostring_rgb(), dtype='uint8')
        prob_images.append(prob_image.reshape(prob_fig.canvas.get_width_height()[::-1] + (3,)))

        data_image = np.frombuffer(data_fig.canvas.tostring_rgb(), dtype='uint8')
        data_images.append(data_image.reshape(data_fig.canvas.get_width_height()[::-1] + (3,)))

        post_image = np.frombuffer(post_fig.canvas.tostring_rgb(), dtype='uint8')
        post_images.append(post_image.reshape(post_fig.canvas.get_width_height()[::-1] + (3,)))

    imageio.mimsave('mean_changepoint_probability.gif', prob_images, fps=30, subrectangles=True)
    imageio.mimsave('mean_data_stream.gif', data_images, fps=30, subrectangles=True)
    imageio.mimsave('mean_posterior_distribution.gif', post_images, fps=30, subrectangles=True)


if __name__ == '__main__':
    make_gifs()
