import torch
from torch import nn
import matplotlib.pyplot as plt

# 1.

device = "cuda" if torch.cuda.is_available() else "cpu"

start = 0
end = 1
step = 0.02

weight = 0.3
bias = 0.9

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + weight * X

train_part = int(0.8 * len(X))
X_train, y_train = X[:train_part], y[:train_part]
X_test, y_test = X[train_part:], y[train_part:]


def plot_predictions(train_data=X_train, train_label=y_train,
                     test_data=X_test, test_label=y_test, predictions=None):

    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_label, s=4, c="red", label="Train data")
    plt.scatter(test_data, test_label, s=4, c="blue", label="Test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, s=4, c="green", label="Predictions")

    plt.legend(prop={"size": 14})

    plt.show()
    plt.close()


plot_predictions()

# 2.
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_params = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_params(x)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

torch.manual_seed(0)
model = LinearRegressionModel()
model.to(device)
print(model.state_dict())


# 3.
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


def training():
    torch.manual_seed(0)
    epochs = 300
    for epoch in range(epochs):
        # Train
        model.train()
        y_preds = model(X_train)
        loss = loss_fn(y_preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Test
        model.eval()
        with torch.inference_mode():
            global test_preds
            test_preds = model(X_test)
            test_loss = loss_fn(test_preds, y_test)

        if epoch % 20 == 0:
            print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")


training()
# 4.
with torch.inference_mode():
    new_y_preds = model(X_test)

plot_predictions(predictions=new_y_preds.cpu())
from pathlib import Path
MODEL_PATH = Path("model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "model.pt"
MODEL_PATH_COMPLETE = MODEL_PATH / MODEL_NAME

torch.save(obj=model.state_dict(), f=MODEL_PATH_COMPLETE)

# 5.
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(MODEL_PATH_COMPLETE))
loaded_model = loaded_model.to(device)

with torch.inference_mode():
    loaded_preds = loaded_model(X_test)

print(loaded_preds == test_preds)
print(X_train.shape)