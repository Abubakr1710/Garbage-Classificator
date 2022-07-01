import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from my_model import model
from data_handler import dataloader
torch.manual_seed(0)

trainloader, testloader = dataloader(pth='dataset')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

num_epochs = 30
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

loss_train = []
loss_test = []
test_acc= []
b_acc = 0.85
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    

    loss_epoch = 0
    # training
    for tr_images, tr_labels in trainloader:
        tr_images = tr_images.to(device)
        tr_labels = tr_labels.to(device)

        optimizer.zero_grad()
        output = model(tr_images)
        train_preds = torch.argmax(output.detach(), dim=1)
        train_loss = criterion(output, tr_labels)
        loss_epoch += train_loss.item()

        train_loss.backward()
        optimizer.step()

    loss_train.append(loss_epoch / len(trainloader))

    print(f'Train loss: {loss_train[-1]}')

    model.eval()
    with torch.no_grad():
        acc = 0
        test_loss_epoch = 0

        for te_images, te_labels in testloader:
            te_images = te_images.to(device)
            te_labels = te_labels.to(device)
            test_output = model(te_images)
            test_preds = torch.argmax(test_output, dim=1)
            test_accuracy = (test_preds == te_labels).sum() / len(te_labels)
            acc += test_accuracy.item()
            test_losses = criterion(test_output, te_labels)
            test_loss_epoch += test_losses.item()

        test_acc.append(acc / len(testloader))
        loss_test.append(test_loss_epoch / len(testloader))

        print(f'Test accuracy: {test_acc[-1]*100 :.2f}%')
        if b_acc < test_acc[-1]:
            torch.save(model.to('cpu').state_dict(), './model.pth')
            model.to(device)
            b_acc = test_acc[-1]

    model.train()


# Plots
x_epochs = list(range(num_epochs))
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(x_epochs, loss_train)
plt.plot(x_epochs, loss_test)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_epochs, test_acc)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('result')

plt.show()