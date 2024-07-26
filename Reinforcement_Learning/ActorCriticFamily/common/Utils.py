import matplotlib.pyplot as plt

def save_learning_plot(x, scores, path_to_save):
    # running_avg = np.zeros(len(scores))
    # for i in range(len(running_avg)):
    #     running_avg[i] = np.mean(scores[max(0, i-100):(i+101)])
    plt.cla()
    plt.plot(x, scores)
    plt.title('Learning Curve')
    plt.savefig(path_to_save)
