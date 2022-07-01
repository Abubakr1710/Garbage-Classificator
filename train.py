import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from my_model import model
from data_handler import dataloader
torch.manual_seed(0)

device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
traindata, testdata = dataloader(pth = 'dataset')

def torch_fit(num_epochs, traindata, testdata, model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    tr_loss = []
    te_loss = []
    acc_train = []
    acc_test = []
    b_acc = 0.85

    for epoch in range(num_epochs):
        print(f'Epoch{epoch+1}/{num_epochs}')

        tr_epoch =[]
        for i, (tr_images, tr_labels) in enumerate(traindata):
            tr_images = tr_images.to(device)
            tr_labels = tr_labels.to(device)

            optimizer.zero_grad()
            tr_output = model(tr_images)
            tr_pred = torch.argmax(tr_output.detach(), dim=1)
            train_loss = criterion(tr_output, tr_labels)
            tr_epoch.append(train_loss.item())

        model.eval()
        with torch.no_grad():

            te_epoch = []
            acc_on_epoch = []
            for j, (te_images, te_labels) in enumerate(testdata):
                te_images = te_images.to(device)
                te_labels = te_labels.to(device)

                te_output = model(te_images)
                te_preds = torch.argmax(te_output, dim=1)
                test_loss = criterion(te_output, te_labels)
                te_epoch.append(test_loss.item())
                test_acc = (te_preds == te_labels).sum() / len(te_labels)
                acc_on_epoch.append(test_acc)

        mean_acc =sum(acc_on_epoch)/len(acc_on_epoch)
        acc_test.append(mean_acc)

        test_loss_mean = sum(te_epoch)/len(te_epoch)
        te_loss.append(test_loss_mean)

        train_loss_mean = sum(tr_epoch)/len(tr_epoch)
        tr_loss.append(train_loss_mean)

        if b_acc < mean_acc:
            torch.save(model.state_dict(),'checkpoint.pth')
            state_dict = torch.load('checkpoint.pth')
            #print(state_dict.keys())
            b_acc = mean_acc

        model.train()

        print(f'Mean epoch loss for train: {train_loss_mean}')
        print(f'Mean epoch loss for test: {test_loss_mean}')
        print(f'accuracy on epoch: {mean_acc}')

    x_axis_acc=list(range(num_epochs))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(tr_loss, label='Train loss')
    plt.plot(te_loss, label='Test loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(x_axis_acc,acc_test, label='Accuracy')
    plt.legend()
    plt.show()

ans = torch_fit(num_epochs=30, traindata=traindata, testdata=testdata, model=model)