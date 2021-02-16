import torch 
from torch import nn


class VAE(nn.Module):
    def __init__(self, categorical_dim=2, 
                       categorical_latent_dim=1,
                       continous_latent_dim=1):

        self.categorical_latent_dim = categorical_latent_dim
        self.categorical_dim = categorical_dim
        self.continous_latent_dim = continous_latent_dim

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, categorical_latent_dim*categorical_dim)

        self.fc4 = nn.Linear(continous_latent_dim, 256)
        self.fc5 = nn.Linear(256,512)
        self.fc6 = nn.Linear(512,784)

        self.fc7 = nn.Linear(2*categorical_latent_dim*categorical_dim,2*continous_latent_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def encode_continous(self,x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def combine_z_qc(self, z, q_c):
        combined = torch.cat([z,q_c], dim=1)
        return combined

    def forward(self, x,temp):
        # Discrete Encoder
        q = self.encode(x.view(-1, 784))
        q_y = q.view(q.size(0),self.latent_dim,self.categorical_dim)
        z = gumbel_softmax(q_y,temp)

        # Continous encoder
        q_c = self.encode_continous(x.view(-1, 784))
        combined = self.combine_z_qc(z, q_c)
        q_yc = self.fc7(combined.view(-1, 2*self.categorical_latent_dim*self.categorical_dim))
        z_continous = sample_gaussian(q_yc, self.continous_latent_dim)
        return self.decode(z)

def sample_gaussian(q_yc, continous_latent_dim):
    mean = q_yc[:continous_latent_dim]
    sig = torch.exp(q_yc[continous_latent_dim:])
    z_continous = torch.normal(mean, sig)
    return z_continous

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1,latent_dim*categorical_dim)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    log_qy = torch.log(qy+1e-20)
    g = Variable(torch.log(torch.Tensor([1.0/categorical_dim])).cuda())
    KLD_d = torch.sum(qy*(log_qy - g),dim=-1).mean()
    KLD_c = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD_d + KLD_c


