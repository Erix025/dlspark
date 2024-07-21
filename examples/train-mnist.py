import dlspark
import dlspark.nn as nn
import numpy as np
from dlspark.auto_grad import Tensor
from tqdm.auto import tqdm

# from dlspark.nn.optimizer import Adam# Hyperparameters
batch_size = 50
epochs = 10
workers = 16
learning_rate = 0.01
momentum = 0.9
step_size = 5
step_gamma = 0.5

# Load MNIST dataset
train_set = dlspark.data.datasets.MNISTDataset(dataset_path='./datasets',
                                       train = True)
test_set = dlspark.data.datasets.MNISTDataset(dataset_path='./datasets',
                                      train = False)

train_loader = dlspark.data.DataLoader(train_set,
                                           batch_size = batch_size)
test_loader = dlspark.data.DataLoader(test_set,
                                          batch_size = batch_size)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Training on {}...".format(device))

# # Init tensorboard writer
# writer = SummaryWriter(comment = ' Pytorch GELU - {}'.format(device))

def evaluate(model : dlspark.nn.Module, idx):
    model.eval()
    total_correct = 0
    total_samples = 0

    for x_test, y_test in test_loader:
        outputs = model(x_test)
        # Get index of max value
        y_pred = outputs.data.argmax(1)
        y_pred = y_pred.astype(y_test.dtype)
        total_correct += (y_pred == y_test.data).sum().item()
        total_samples += y_test.shape[0]

    accuracy = total_correct / total_samples

    # print without new line
    tqdm.write('Index: {} Accuracy: {:.2%}'.format(idx, accuracy))



# Definition of LeNet-5
model = nn.Sequential(
    # linear 
    nn.Conv2d(1, 6, 5),         # C1
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(6, 16, 5),  
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), # C5
    nn.ReLU(),
    nn.Linear(120, 84),         # F6
    nn.ReLU(),
    nn.Linear(84, 10),         # F6
    # nn.Linear(84, 10)           # Output Layer
)

# Loss function, optimizer and learning rate scheduler
# criterion = nn.SoftmaxLoss()
criterion = dlspark.nn.loss.CrossEntropyLoss()
# optimizer = Adam(model.parameters(), amsgrad=True, lr=learning_rate)
# optimizer = dlspark.nn.optimizer.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = dlspark.nn.optimizer.Adam(model.parameters(), amsgrad=True, lr=learning_rate)

# Train & Evaluate
for epoch in tqdm(range(epochs)):
    with tqdm(total=train_loader.__len__(), desc=f'Epoch {epoch+1}/{epochs}', unit='batches') as pbar:
        for i, data in enumerate(train_loader):
            # Clear gradient information
            optimizer.zero_grad()
            
            # Forward pass
            x_test, y_test = data
            y_pred = model(x_test)
            loss = criterion(y_pred, y_test)

            # Backward
            loss.backward()
            optimizer.step()
            pbar.update(1)
            idx = epoch * len(train_loader) + i
            if idx % 100 == 0:
                evaluate(model, idx)