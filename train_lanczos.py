# -*- coding: utf-8 -*-
"""VGG CIFAR Hessian updated.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1thL21_zjDfPG3MpCY2kdI0CmKErgWNHH
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse
# Create a normalized random vector
import time
import gpytorch
import os

parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage.')
# Positional argument

# Optional argument
# parser.add_argument('--output', type=str, help='Output file path')
parser.add_argument('--model', type=str, help='Output file path', default="vgg")
# Flag (boolean) argument
parser.add_argument('--augment', action='store_true', help='Increase output verbosity')
parser.add_argument('--noise', type=float, default=0, help='Scale factor')
parser.add_argument('--epochs', type=int, help='A list of numbers', default=10)
parser.add_argument('--batch_size', type=int, help='A list of numbers', default=10)

parser.add_argument('--lr', type=float, default=1e-3, help='Scale factor')
parser.add_argument('--lanczos_beta', type=float, default=1, help='Scale factor')
parser.add_argument('--momentum', type=float, default=0.9, help='Scale factor')
parser.add_argument('--wd', type=float, default=0.0005, help='Scale factor')

parser.add_argument('--dataset_a', type=int, nargs='+', help='A list of numbers')
parser.add_argument('--dataset_b', type=int, nargs='+', help='A list of numbers')

# Optional argument with default value
args = parser.parse_args()



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def _bn_train_mode(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.train()

def hess_vec(vector, loader, model, criterion, cuda=True, bn_train_mode=False):
    param_list = list(model.parameters())
    vector_list = []

    offset = 0
    for param in param_list:
        vector_list.append(vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    model.eval()
    if bn_train_mode:
        model.apply(_bn_train_mode)

    model.zero_grad()
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)
        loss *= input.size()[0] / N

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
        dL_dvec = torch.zeros(1, device='cuda' if cuda else 'cpu')
        for v, g in zip(vector_list, grad_list):
            dL_dvec += torch.sum(v * g)
        dL_dvec.backward()

    model.eval()
    return torch.cat([param.grad.view(-1) for param in param_list]).view(-1)



class CurvVecProduct(object):
    def __init__(self, loader, model, criterion, init_vec=None):
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.init_vec = init_vec
        self.iters = 0
        self.timestamp = time.time()

    def __call__(self, vector):
        if self.iters == 0 and self.init_vec is not None:
            vector = self.init_vec
        start_time = time.time()
        output = hess_vec(
            vector,
            self.loader,
            self.model,
            self.criterion,
            cuda=True,
            bn_train_mode=True
        )
        time_diff = time.time() - start_time
        self.iters += 1
        print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        # return output.cpu().unsqueeze(1)
        return output.unsqueeze(1)



# Now, add the custom noise transformation to your existing pipeline

if args.augment:
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        AddGaussianNoise(0., args.noise),  # Add Gaussian noise with a mean of 0 and std of 0.1
    ])
else:
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        AddGaussianNoise(0., args.noise),
    ])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform_test)

output_folder = 'output/lanczos/{}/{}/lr={}_wd={}_batchsize={}_beta={}/'.format(args.model,"_".join([str(s) for s in args.dataset_a]),str(args.lr),str(args.wd),str(args.batch_size),str(args.lanczos_beta))
try:
    os.makedirs(output_folder)
except:
    pass
print('output folder = {}'.format(output_folder))
# Function to get a subset of the dataset for a specific class
def get_class_subset(dataset, class_list):
    indices = [i for i, (_, label) in enumerate(dataset) if (label in class_list)]
    subset = Subset(dataset, indices)
    return subset

def create_dataset(class_list):
    dataset_train = get_class_subset(train_set, class_list)  # Class 0 is zerofour
    dataset_test = get_class_subset(test_set, class_list)  # Class 0 is zerofour

    # Create DataLoaders for the zerofour class for both training and testing
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    return loader_train, loader_test

a_train, a_test = create_dataset(args.dataset_a)
b_train, b_test = create_dataset(args.dataset_b)

if args.model == "vgg":
    print('running VGG model')

    # Define the ResNet-50 model without pre-trained weights
    model = models.vgg16(pretrained=False)
    # Modify the last fully connected layer to match the number of classes in CIFAR-10
    # Modify the last layer of the classifier to match the number of classes in CIFAR-10
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)
elif args.model == "resnet":
    print('running ResNet model')
    model = models.resnet50(pretrained=False)
    # Modify the last fully connected layer to match the number of classes in CIFAR-10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

model = model.to('cuda')

# Define a loss function and optimizer
print('using cross entropy loss')
criterion = nn.CrossEntropyLoss()
# Assuming zerofour_loader_train is defined elsewhere and is your DataLoader
total_steps = len(a_train) * args.epochs  # Number of batches * number of epochs

# Define a learning rate scheduler for linear decay
def linear_decay(step):
    return max(0, 1 - step / total_steps)


# Initial learning rate and momentum
lr = args.lr
momentum = args.momentum
weight_decay = args.wd

# Initialize momentum buffers for each parameter
momentum_buffers = {}
for param in model.parameters():
    momentum_buffers[param] = torch.zeros_like(param.data)

# Train the network for 10 epochs using the zerofour_loader_train
for epoch in range(args.epochs):  # Loop over the dataset multiple times if needed
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(a_train):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')  # Move inputs and labels to GPU

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Compute gradients manually
        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        gradients.cuda()

        P = sum(p.numel() for p in model.parameters())
        grad_vector = torch.cat([grad.view(-1) for grad in gradients])
        grad_vector.cuda()
        # grad_vector = grad_vector.cuda()
        adjusted_grad_vector = grad_vector.clone()  # Initialize adjusted gradient vector

        grad_vector = grad_vector / torch.norm(grad_vector)

        # Pass the random vector as the initial vector to the CurvVecProduct
        productor = CurvVecProduct(a_train, model, criterion, init_vec=grad_vector)

        # Run the Lanczos algorithm
        lanczos_iters = 15
        Q, T = gpytorch.utils.lanczos.lanczos_tridiag(
            productor,
            max_iter=lanczos_iters,
            dtype=torch.float32,
            device='cuda',
            matrix_shape=(P, P)
        )

        eigvals, eigvects = torch.linalg.eigh(T)
        gammas = eigvects[0, :] ** 2
        V = eigvects.t() @ Q.t()
        delta = args.lanczos_beta


        # Compute adjustments based on eigenvalues and eigenvectors
        for i, eigval in enumerate(eigvals):
            vi = V[:, i]  # i-th eigenvector
            dot_product = torch.dot(grad_vector, vi)
            adjustment = (1/eigval - 1/(eigval+delta)) * dot_product * vi
            adjusted_grad_vector += adjustment

        # Convert the split tensors to the correct shape
        adjusted_gradients = [g.view(p.size()) for g, p in zip(adjusted_gradients, model.parameters())]

        # Perform the manual SGD update with momentum, using adjusted gradients
        with torch.no_grad():
            for param, adj_grad in zip(model.parameters(), adjusted_gradients):
                # Compute the 'flattened' version of the weight decay term + adjusted gradient
                weight_decay_term = weight_decay * param.data if weight_decay != 0 else 0
                adjusted_grad_with_weight_decay = adj_grad + weight_decay_term

                # Update the momentum buffer with the adjusted gradient
                if param in momentum_buffers:
                    momentum_buffers[param] = momentum_buffers[param] * momentum + adjusted_grad_with_weight_decay
                else:
                    momentum_buffers[param] = adjusted_grad_with_weight_decay

                # Update the parameter
                param.data -= lr * momentum_buffers[param]

        # Update the learning rate
        lr = linear_decay(epoch * len(a_train) + i) * args.lr

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # Print every 10 mini-batches
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10:.4f}, Accuracy: {100 * correct / total:.2f}%, LR: {lr:.6f}')
            running_loss = 0.0
            correct = 0
            total = 0

print('Finished Training')

# Specify the path to save the model
model_save_path = output_folder+'/entire_model.pth'

# Save the entire model
torch.save(model, model_save_path)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in a_train:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test {} images: %d %%'.format(" ".join([str(s) for s in args.dataset_a])) % (
    100 * correct / total))
model.train()

# Set the model to evaluation mode
model.eval()

correct = 0
total = 0
total_loss = 0.0  # Initialize total loss

with torch.no_grad():
    for data in b_test:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)

        loss = criterion(outputs, labels)  # Compute the loss
        total_loss += loss.item() * images.size(0)  # Update total loss

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate average loss
average_loss = total_loss / total

print(f'Accuracy of the network on the test {" ".join([str(s) for s in args.dataset_b])} images: {100 * correct / total:.2f}%')
print(f'Average loss on the test {" ".join([str(s) for s in args.dataset_b])} images: {average_loss:.4f}')

print('running Lanczos on original training set model')

P = sum(p.numel() for p in model.parameters())
random_vec = torch.randn(P, device='cuda')
random_vec = random_vec / torch.norm(random_vec)

# Pass the random vector as the initial vector to the CurvVecProduct
productor = CurvVecProduct(a_train, model, criterion, init_vec=random_vec)

# Run the Lanczos algorithm
lanczos_iters = 20
Q, T = gpytorch.utils.lanczos.lanczos_tridiag(
    productor,
    max_iter=lanczos_iters,
    dtype=torch.float32,
    device='cuda',
    matrix_shape=(P, P)
)

eigvals, eigvects = torch.linalg.eigh(T)
eigvals = eigvals

gammas = eigvects[0, :] ** 2
V = eigvects.t() @ Q.t()

# Save or return the results as needed
result = {
    # 'w': w,
    'eigvals': eigvals,
    'gammas': gammas,
    'V': V
}
print(eigvals)
checkpoint_path = output_folder+'eigenspace.pth'

# Save the result dictionary
torch.save(result, checkpoint_path)

print(f'Results saved to {checkpoint_path}')
