import subprocess

poison_rates = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]

data_path = "./CIFAR10_dataset"
batch_size = 128
epochs = 20
learning_rate = 0.001
seed = 1
log_interval = 10

for poison_rate in poison_rates:
    print(
        f"******************* CIFAR-10 EXPERIMENT | PoisonRate={poison_rate} | Epochs={epochs} *******************"
    )

    command = [
        "python", "cifar10_train.py",
        "--data", data_path,
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(learning_rate),
        "--seed", str(seed),
        "--poison_rate", str(poison_rate),
        "--log-interval", str(log_interval),
    ]

    subprocess.run(command)
