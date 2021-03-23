"""Jupyter notebook 에서 사용하는 함수
    * 학습, 예측 결과 검증용

"""

from matplotlib import pyplot as plt


def plot_performance(epoch, train_history:list, validation_history:list, target:str):

    fig = plt.figure(figsize=(12, 5))
    epoch_range = list(range(epoch))

    plt.plot(epoch_range, train_history, marker='.', c='red', label="train")
    plt.plot(epoch_range, validation_history, marker='.', c='blue', label="validation")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel(target)

    return fig


if __name__ == '__main__':
    pass