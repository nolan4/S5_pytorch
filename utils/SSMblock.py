import torch

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
        activations = torch.nn.GELU(preactivations)
        return activations

# # Example usage
# P = 5 # latent dimension
# H = 9  # num features
# ssm_block = SSMblock(P, H)

# L = 25 # <-- input sequence length
# B = 5 # Batch size
# dummy_input = torch.randn(B, L, H)  # Assuming input shape matches expectations
# output = ssm_block(dummy_input)