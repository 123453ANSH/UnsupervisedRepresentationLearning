import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader
from data import RotDataset
from resnet import ResNet
import time
import shutil
import yaml
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_epoch', type=int, required=False)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    progress_bar = tqdm(train_loader)
    for i, (input, target) in enumerate(progress_bar):
    	#TODO: use the usual pytorch implementation of training
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        N = outputs.shape[0]
        softmaxed = nn.Softmax(dim = 1)(outputs)
        one_hot_labels = torch.zeros(outputs.shape)
        one_hot_labels[torch.arange(N), target] = 1
        accuracy = 1 - (torch.sum(torch.abs(softmaxed - one_hot_labels))/(2*N))

        progress_bar.set_description("Accuracy: " + str(accuracy.item()) + '  |  Loss: ' + str(torch.sum(loss).item()))

def validate(val_loader, model, criterion):
    model.eval()
    losses = []
    accuracies = []
    progress_bar = tqdm(val_loader)

    for i, (input, target) in enumerate(progress_bar):
        outputs = model(input)
        N = outputs.shape[0]
        loss = criterion(outputs, target)
        losses.append(loss.item())

        softmaxed = nn.Softmax(dim = 1)(outputs)
        one_hot_labels = torch.zeros(outputs.shape)
        one_hot_labels[torch.arange(N), target] = 1
        accuracy = 1 - (torch.sum(torch.abs(softmaxed - one_hot_labels))/(2*N))

        accuracies.append(accuracy)

    #returns mean of loss and accuracy over all training examples
    batches = i + 1
    return [sum(losses)/batches, sum(accuracies)/batches]


def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
    torch.save(state, filename)
    #best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

def main():
    n_epochs = config["num_epochs"]
    model = ResNet(None, None, 4)

    if args.resume:
        checkpoint = torch.load('./rotationnetcheckpoint.pth.tar')
        model.load_state_dict(checkpoint)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                           momentum = config['momentum'],
                           lr = config['learning_rate'],
                           weight_decay = config['weight_decay'])

    train_dataset = RotDataset(args.data_dir + '/train', shuffle = True)
    train_loader = DataLoader(train_dataset, batch_size = config['batch_size'])
    val_dataset = RotDataset(args.data_dir + '/test')
    val_loader = DataLoader(val_dataset, batch_size = config['batch_size'])

    best_loss = 10e10

    for epoch in range(args.resume_epoch, n_epochs):
        #TODO: make your loop which trains and validates. Use the train() func
        print("============================\nEPOCH " + str(epoch) + '\n')
        if args.train:
            train(train_loader, model, criterion, optimizer, epoch)
            print("Epoch " + str(epoch) + " finished training")

        loss, accuracy = validate(val_loader, model, criterion)
        print("Batch Loss: " + str(loss) + "  |   Batch Accuracy: " + str(accuracy))

        best = False
        if loss < best_loss:
            best = True
            best_loss = loss

        save_checkpoint(model.state_dict(), best)

        print('============================')



if __name__ == "__main__":
    main()
