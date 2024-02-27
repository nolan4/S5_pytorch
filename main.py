import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F


class SSMblock(torch.nn.Module):
    def __init__(self, P, H, delta_min=0.001, delta_max=0.1):
        super(SSMblock, self).__init__()
        
        self.P = P # latent dimension
        self.H = H # input channels
        
        # TODO: fancy initializaiton of A using HiPPO or LegS etc.
        A = torch.nn.Parameter(torch.randn(P, P, dtype=torch.float).to(torch.complex64)) 

        # eigendecomposition of A to get eigenvector matrix (V) and eigenvector diagonal matrix (Lambda)
        eigvals_A, eigvecs_A = torch.linalg.eig(A)
        self.Lambda = eigvals_A
        
        # initialize B~ and C~ using eigenvector matrix (V)
        self.B_tilde = torch.linalg.inv(eigvecs_A) @ torch.nn.Parameter(torch.randn(P, H, dtype=torch.float).to(torch.complex64))
        self.C_tilde = torch.nn.Parameter(torch.randn(H, P, dtype=torch.float).to(torch.complex64)) @ torch.linalg.inv(eigvecs_A)
        
        # initialize D (not used)
        self.D = torch.nn.Parameter(torch.randn(H, H, dtype=torch.float).to(torch.complex64))
        
        # Initialize log(Δ) from a uniform distribution over [log(δ_min), log(δ_max))
        self.delta = torch.nn.Parameter(torch.log(torch.rand(P) * (delta_max - delta_min) + delta_min))
        self.I = torch.eye(P) # Identity matrix, ensure complex dtype to match other parameters
        
    def hippoLegsInit():
        return None
        
    def discretize(self, Lambda, B_tilde, delta):
        Lambda_ = torch.exp(Lambda * self.delta)
        B_ = (1 / Lambda * (Lambda_ - self.I)) @ B_tilde
        return Lambda_, B_
            
    
    def apply_SSM(self, Lambda_, B_, input_sequence):
        # Adjust for batched input: [B, L, H]
        B, L, H = input_sequence.shape

        # Ensure input_sequence is complex
        input_sequence_complex = input_sequence.to(torch.complex64)

        # Pre-compute Lambda diagonal matrix
        Lambda_diag = torch.diag_embed(Lambda_)

        # Initialize states tensor for all batches
        states = torch.zeros((B, L, self.P), dtype=torch.complex64, device=input_sequence.device)

        # Compute states iteratively for each batch
        for b in range(B):
            for i in range(L):
                state_update = B_ @ input_sequence_complex[b, i] if i == 0 else Lambda_diag @ states[b, i-1] + B_ @ input_sequence_complex[b, i]
                states[b, i] = state_update.squeeze()

        # Compute outputs from states for all batches
        ys_complex = torch.einsum('hj,bij->bih', self.C_tilde, states)  # Use einsum for batched matmul

        # Add feedthrough contribution from input_sequence and take the real part
        ys = (ys_complex + torch.einsum('hj,bij->bih', self.D, input_sequence_complex)).real

        return ys
        
            
    def forward(self, x):
        # <----------------------
        # discretize Lambda and B~. From the paper, C_ = C~ and D_ = D
        Lambda_, B_ = self.discretize(self.Lambda, self.B_tilde, torch.exp(self.delta))
        # <----------------------
        preactivations = self.apply_SSM(Lambda_, B_, x)
        # <----------------------
        activations = F.gelu(preactivations)
        return activations


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.ssm_block1 = SSMblock(P=32, H=1)  # P -> latent dimension | H -> channels
        self.ssm_block2 = SSMblock(P=16, H=1)  # P -> latent dimension | H -> channels
        self.linear = nn.Linear(784, 10)
        
    def forward(self, x):
        
        print(f'x.shape from dataloader:', x.shape)
        
        x = torch.flatten(x, start_dim=2)  # Flatten the 28x28 part
        x = x.view(x.size(0), -1, 1)  # Reshape to [batch, 784, 1]
        
        print(f'x.shape after flattening batch:', x.shape)
        
        x = self.ssm_block1(x)
        
        print(f'x.shape after ssmblock1:', x.shape)
        
        x = self.ssm_block2(x)
        
        print(f'x.shape after ssmblock2:', x.shape)
        x = torch.flatten(x, start_dim=1)  # This will flatten the output to [batch, 784]
        print(f'x.shape after flatten:', x.shape)
        x = self.linear(x)
        
        print(f'x.shape after linear:', x.shape)
        
        return x
        

def train_model(model, train_loader, test_loader, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):  # Loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    print('Finished Training')

def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    torch.autograd.set_detect_anomaly(True)

    # Load MNIST dataset
    dataset_path = './dataset'
    train_dataset = MNIST(root=dataset_path, train=True, transform=ToTensor(), download=True)
    test_dataset = MNIST(root=dataset_path, train=False, transform=ToTensor())

    # Prepare data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4)

    # Initialize the model
    model = MyModel()

    # Train the model
    train_model(model, train_loader, test_loader, device)

if __name__ == '__main__':
    main()