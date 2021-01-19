import matplotlib.pyplot as plt


def draw(every_epoch_mean_train_losses, every_epoch_mean_valid_losses):
    current_epochs = len(every_epoch_mean_valid_losses)
    ax = plt.subplot(1, 1, 1)
    ax.plot(list(range(1, 1 + current_epochs)), every_epoch_mean_train_losses, label="train loss", color="r")
    ax.plot(list(range(1, 1 + current_epochs)), every_epoch_mean_valid_losses, label="valid loss", color="g")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    with open("./every_epoch_mean_train_losses.txt", "r", encoding="utf-8") as file:
        every_epoch_mean_train_losses = eval(file.read())
    with open("./every_epoch_mean_valid_losses.txt", "r", encoding="utf-8") as file:
        every_epoch_mean_valid_losses = eval(file.read())
    draw(every_epoch_mean_train_losses, every_epoch_mean_valid_losses)