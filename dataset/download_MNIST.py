import torchvision
from torchvision.transforms import ToTensor

# Define the root directory where the dataset will be stored
root = './dataset'

# Download and load the training data
train_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=ToTensor(), download=True)

# Download and load the test data
test_dataset = torchvision.datasets.MNIST(root=root, train=False, transform=ToTensor(), download=True)
