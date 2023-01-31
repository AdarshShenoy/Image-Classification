# Imports here
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

# Getting Arguments from CLI
parser = argparse.ArgumentParser()
parser.add_argument('--arch', dest='arch', type=str, default='vgg16', choices=['vgg16', 'alexnet'])
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default='0.001')
parser.add_argument('--hidden_inputs', type=int, dest='hidden_inputs', default=4096)
parser.add_argument('--epochs', type=int, dest='epochs', default=8)
parser.add_argument('--gpu', type=str, action='store', default='cpu')
arguments = parser.parse_args()
args = parser.parse_args()

# Directories
train_directory = 'flowers' + '/train'
valid_directory = 'flowers' + '/valid'
test_directory = 'flowers' + '/test'

##Downloading the model
model = getattr(models, args.arch)(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

if args.arch == 'vgg16':
    classifier = nn.Sequential(OrderedDict([('Layer1', nn.Linear(25088, args.hidden_inputs)),
                                            ('Activation1', nn.ReLU()),
                                            ('Dropout', nn.Dropout(0.2)),
                                            ('Layer2', nn.Linear(args.hidden_inputs, 1024)),
                                            ('Activation2', nn.ReLU()),
                                            ('Layer3', nn.Linear(1024, 512)),
                                            ('Activation3', nn.ReLU()),
                                            ('Layer4', nn.Linear(512, 102)),
                                            ('Output', nn.LogSoftmax(dim=1))]))

elif args.arch == 'alexnet':
    classifier = nn.Sequential(OrderedDict([('Layer1', nn.Linear(1000, args.hidden_inputs)),
                                            ('Activation1', nn.ReLU()),
                                            ('Dropout', nn.Dropout(0.2)),
                                            ('Layer2', nn.Linear(args.hidden_inputs, 102)),
                                            ('Output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier

if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'


def transform(train_dir, valid_dir, test_dir):
    ##Creating Transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the transforms, defining the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    return train_loader, validation_loader, test_loader


def validation(model, valLoader, criterion, device):
    val_loss = 0
    accuracy = 0
    for images, labels in iter(valLoader):
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        # Probability distribution
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return val_loss, accuracy


def train_classifier(model, train_loader, validation_loader, epochs, criterion, optimizer, device):
    steps = 0
    print_every = 20
    model.to(device)

    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in iter(train_loader):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward pass
            output = model.forward(images)
            # Calculate loss
            loss = criterion(output, labels)
            # Backpropagation
            loss.backward()
            # Optimization
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validation_loader, criterion, device)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss / len(validation_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(validation_loader))
                      )
                running_loss = 0
                model.train()


def test_accuracy(model, test_loader, device):
    model.eval()
    model.to(device)

    with torch.no_grad():
        accuracy = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            #Probability distribution
            ps = torch.exp(output)
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        print("Test Accuracy: {:.3f} %".format(100 * accuracy / len(test_loader)))


def save_checkpoint(model, train_dataset, optimizer):
    model.class_to_idx = train_dataset.class_to_idx
    torch.save({
        'arch': args.arch,
        'model': model,
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': classifier,
    }, 'checkpoint.pth')
    print("Saved")


def main():
    train_loader, validation_loader, test_loader = transform(train_directory, valid_directory, test_directory)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    train_classifier(model, train_loader, validation_loader, args.epochs, criterion, optimizer, device)
    test_accuracy(model, test_loader, device)
    save_checkpoint(model, train_loader, optimizer)


if __name__ == "__main__":
    main()
