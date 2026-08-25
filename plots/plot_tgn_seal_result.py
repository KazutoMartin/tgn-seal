import pickle

import matplotlib.pyplot as plt

AVAILABLE_DATASET = ["dept1", "dept2", "dept3", "dept4"]
DATASET = AVAILABLE_DATASET[1]


def read_file(file_name):
    with open(file_name, "rb") as f:
        d = pickle.load(f)
    f.close()

    return d


file_path = f"./results/tgn-seal-{DATASET}-layered-cache-2hop.pkl"
results = read_file(file_path)

epochs = range(1, len(results['val_aps']) + 1)

# Plot validation APs
plt.figure(figsize=(8, 5))
plt.plot(epochs, results['val_aps'], label="Validation AP (Seen nodes)")
plt.plot(epochs, results['new_nodes_val_aps'], label="Validation AP (New nodes)")
plt.xlabel("Epoch")
plt.ylabel("Average Precision (AP)")
plt.title("Validation Performance Over Epochs")
plt.legend()
plt.xticks(epochs)
plt.show()

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, results['train_losses'], color="red")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Epochs")
plt.xticks(epochs)
plt.show()
