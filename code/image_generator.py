# Simple GAN Example (PyTorch)

import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize
generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(100):
    # Train Discriminator
    real_data = torch.randn(10, 1)
    fake_data = generator(torch.randn(10, 10)).detach()

    real_labels = torch.ones(10, 1)
    fake_labels = torch.zeros(10, 1)

    d_loss_real = criterion(discriminator(real_data), real_labels)
    d_loss_fake = criterion(discriminator(fake_data), fake_labels)
    d_loss = d_loss_real + d_loss_fake

    optimizer_d.zero_grad()
    d_loss.backward()
    optimizer_d.step()

    # Train Generator
    noise = torch.randn(10, 10)
    generated_data = generator(noise)

    g_loss = criterion(discriminator(generated_data), real_labels)

    optimizer_g.zero_grad()
    g_loss.backward()
    optimizer_g.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")
