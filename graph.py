import pickle

import matplotlib.pyplot as plt


# data_dic has following key:
# 1. kld
# 2. entropy
# 3. bleu
# 4. kld_weight
# 5. tf_ratio
# 6. gau
def draw_figure(points: int, data_dic: dict):
    fig, ax1 = plt.subplots()
    plt.title('Training loss/ratio curve')
    plt.xlabel('500 iteration(s)')

    x_axis = range(points)

    ax1.set_ylabel('Loss')
    ax1.plot(x_axis, data_dic['kld'], label='KLD', linewidth=2)
    ax1.plot(x_axis, data_dic['entropy'], label='CrossEntropy', linewidth=2)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('score/weight')
    ax2.plot(x_axis, data_dic['bleu'], 'ro', label='BLEU4-score', color='green')
    ax2.plot(x_axis, data_dic['kld_weight'], label='KLD_weight', color='red', linestyle='--')
    ax2.plot(x_axis, data_dic['tf_ratio'], label='Teacher ratio', color='purple', linestyle='--')
    ax2.plot(x_axis, data_dic['gau'], 'ro', label='Gaussian Score', color='brown')
    ax2.tick_params(axis='y')

    # fig.tight_layout()
    fig.legend()
    # plt.show()
    plt.savefig('train_figure1.png')


if __name__ == "__main__":
    dic = {
        'kld': [100, 200, 250, 270, 350],
        'entropy': [2.5, 1.7, 1.3, 1.2, 0.9],
        'bleu': [0.02, 0.34, 0.52, 0.44, 0.62],
        'kld_weight': [0.2, 0.4, 0, 0.1, 0.3],
        'tf_ratio': [1, 1, 1, 1, 1],
        'gau': [0, 0, 0.05, 0.12, 0.25]
    }
    draw_figure(5, dic)

    # f = open('test', 'wb')
    # pickle.dump(dic, f)
    # f.close()

    # f = open('test', 'rb')
    # output = pickle.load(f)
    # print(output)
    # f.close()