import torch
import torch.nn as nn
from data_loader import load_and_clean_data
from models import TransformerGenerator, Discriminator

# Setup
train_loader, input_dim = load_and_clean_data("data/mqttdataset_reduced.csv")
generator = TransformerGenerator(input_dim)
discriminator = Discriminator(input_dim)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

print("Starting Baseline Training...")
for epoch in range(10):
    for i, (real_data, _) in enumerate(train_loader):
        batch_size = real_data.size(0)
        
        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        
        # Real data labels = 1, Fake data labels = 0
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        outputs_real = discriminator(real_data)
        d_loss_real = criterion(outputs_real, real_labels)
        
        fake_data = generator(real_data)
        outputs_fake = discriminator(fake_data.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # --- Train Generator (Adversarial) ---
        optimizer_G.zero_grad()
        # Generator wants Discriminator to think fake data is Real (1)
        outputs_adv = discriminator(fake_data)
        g_loss = criterion(outputs_adv, real_labels)
        
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/10] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

torch.save(generator.state_dict(), "teacher_baseline.pt")
print("Baseline Teacher Model Saved!")