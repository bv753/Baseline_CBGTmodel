import matplotlib.pyplot as plt
import numpy as np


def plot_loss(losses_nm):
    loss_curve_nm = [loss[-1] for loss in losses_nm]
    # loss_curve_vanilla = [loss[-1] for loss in losses_vanilla]
    x_axis = np.arange(len(losses_nm)) * 200
    
    plt.cla()
    plt.plot(x_axis, np.log10(loss_curve_nm), label='NM RNN')
    # plt.plot(x_axis, np.log10(loss_curve_vanilla), label='Vanilla RNN')
    plt.ylabel('log10(error)')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

# continue
