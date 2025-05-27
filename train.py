import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import models.modelv1 as modelv1
from torchsummary import summary

print(f'Running on torch version: {torch.__version__}')
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')

# Load datasets
train_data = dataset.get_train()
test_data = dataset.get_test()

# dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize model
model = modelv1.ModelV1(in_shape=train_data[0][0].shape[0],
                       hidden=64,
                       n_classes=len(train_data.classes)).to(device)

print(f'Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
print(model)
model.to(device)

# optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

# training loop
def train_model(model, epoch):
    model.train()
    for batch, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print(f'Epoch [{epoch}/30], Batch [{batch+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            torch.save(model.state_dict(), 'modelv1.pth')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

for epoch in range(1, 30):
    train_model(model, epoch)
    test()