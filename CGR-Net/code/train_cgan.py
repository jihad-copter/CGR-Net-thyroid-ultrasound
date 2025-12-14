import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
IMG_SIZE = 224  # High resolution for better quality
LATENT_DIM = 100
NUM_CLASSES = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
BETA1 = 0.5
EPOCHS = 500
SAMPLE_INTERVAL = 50

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from class 0 folder
        class0_path = os.path.join(root_dir, 'class_0')
        if os.path.exists(class0_path):
            for img_name in os.listdir(class0_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class0_path, img_name))
                    self.labels.append(0)
        
        # Load images from class 1 folder
        class1_path = os.path.join(root_dir, 'class_1')
        if os.path.exists(class1_path):
            for img_name in os.listdir(class1_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class1_path, img_name))
                    self.labels.append(1)
                    
        print(f"Loaded {len([l for l in self.labels if l == 0])} images for class 0")
        print(f"Loaded {len([l for l in self.labels if l == 1])} images for class 1")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Input dimension is latent_dim + latent_dim (for label embedding)
        # For 224x224, we need more upsampling layers
        self.init_size = img_size // 16  # 224 // 16 = 14
        self.l1 = nn.Sequential(nn.Linear(latent_dim * 2, 512 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),  # 14 -> 28
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 28 -> 56
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 56 -> 112
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 112 -> 224
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Concatenate label embedding with noise
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if normalize:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block
        
        # Input channels: 3 (RGB) + 1 (label channel)
        self.model = nn.Sequential(
            *discriminator_block(4, 32, normalize=False),  # 224 -> 112
            *discriminator_block(32, 64),                  # 112 -> 56
            *discriminator_block(64, 128),                 # 56 -> 28
            *discriminator_block(128, 256),                # 28 -> 14
            *discriminator_block(256, 512),                # 14 -> 7
        )
        
        # Calculate the size after convolution layers: 224 / (2^5) = 7
        ds_size = img_size // 2 ** 5
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Concatenate label embedding as additional channel
        label_input = self.label_emb(labels)
        label_input = label_input.view(labels.shape[0], 1, self.img_size, self.img_size)
        d_in = torch.cat((img, label_input), 1)
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create dataset and dataloader
# Adjust the path to your dataset directory
dataset = CustomDataset(root_dir='path/to/your/dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Initialize models
generator = Generator(LATENT_DIM, NUM_CLASSES, IMG_SIZE).to(device)
discriminator = Discriminator(NUM_CLASSES, IMG_SIZE).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# Training function
def train_cgan():
    print("Starting CGAN training...")
    
    # Create directories for saving samples and models
    os.makedirs("generated_samples", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for i, (real_imgs, labels) in enumerate(progress_bar):
            batch_size = real_imgs.shape[0]
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real images
            real_pred = discriminator(real_imgs, labels)
            d_real_loss = criterion(real_pred, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_labels_input = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
            fake_imgs = generator(noise, fake_labels_input)
            fake_pred = discriminator(fake_imgs.detach(), fake_labels_input)
            d_fake_loss = criterion(fake_pred, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_labels_input = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
            fake_imgs = generator(noise, fake_labels_input)
            
            # Generator loss
            fake_pred = discriminator(fake_imgs, fake_labels_input)
            g_loss = criterion(fake_pred, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            progress_bar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
        
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}] - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}')
        
        # Save sample images
        if (epoch + 1) % SAMPLE_INTERVAL == 0:
            save_sample_images(epoch + 1)
        
        # Save model checkpoints
        if (epoch + 1) % 100 == 0:
            torch.save(generator.state_dict(), f'saved_models/generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'saved_models/discriminator_epoch_{epoch+1}.pth')
    
    print("Training completed!")

def save_sample_images(epoch):
    """Save sample generated images"""
    generator.eval()
    with torch.no_grad():
        # Generate samples for each class
        noise = torch.randn(8, LATENT_DIM).to(device)
        
        # Class 0 samples
        labels_0 = torch.zeros(8, dtype=torch.long).to(device)
        fake_imgs_0 = generator(noise, labels_0)
        
        # Class 1 samples
        labels_1 = torch.ones(8, dtype=torch.long).to(device)
        fake_imgs_1 = generator(noise, labels_1)
        
        # Combine samples
        fake_imgs = torch.cat([fake_imgs_0, fake_imgs_1], 0)
        
        # Save images
        vutils.save_image(fake_imgs, f'generated_samples/epoch_{epoch}.png', 
                         normalize=True, nrow=8)

def generate_augmentation_data():
    """Generate additional data for augmentation"""
    print("Generating augmentation data...")
    
    # Load the trained generator
    generator.load_state_dict(torch.load(f'saved_models/generator_epoch_{EPOCHS}.pth'))
    generator.eval()
    
    os.makedirs("augmented_data/class_0", exist_ok=True)
    os.makedirs("augmented_data/class_1", exist_ok=True)
    
    # Generate 1000 images for class 0
    print("Generating 1000 images for class 0...")
    for i in tqdm(range(1000)):
        with torch.no_grad():
            noise = torch.randn(1, LATENT_DIM).to(device)
            label = torch.zeros(1, dtype=torch.long).to(device)
            fake_img = generator(noise, label)
            
            # Denormalize and save
            fake_img = (fake_img + 1) / 2.0  # Convert from [-1,1] to [0,1]
            vutils.save_image(fake_img, f'augmented_data/class_0/generated_{i:04d}.png')
    
    # Generate 774 images for class 1
    print("Generating 774 images for class 1...")
    for i in tqdm(range(774)):
        with torch.no_grad():
            noise = torch.randn(1, LATENT_DIM).to(device)
            label = torch.ones(1, dtype=torch.long).to(device)
            fake_img = generator(noise, label)
            
            # Denormalize and save
            fake_img = (fake_img + 1) / 2.0  # Convert from [-1,1] to [0,1]
            vutils.save_image(fake_img, f'augmented_data/class_1/generated_{i:04d}.png')
    
    print("Augmentation data generation completed!")

def plot_training_samples():
    """Plot some training samples to verify data loading"""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Training Data Samples')
    
    # Get some samples
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    for i in range(8):
        row = i // 4
        col = i % 4
        img = images[i].permute(1, 2, 0)
        img = (img + 1) / 2.0  # Denormalize
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'Class {labels[i].item()}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_samples.png')
    plt.show()

if __name__ == "__main__":
    # Display training data samples
    plot_training_samples()
    
    # Train the CGAN
    train_cgan()
    
    # Generate augmentation data
    generate_augmentation_data()
    
    print("\nData augmentation process completed!")
    print("Original dataset: Class 0 (1905), Class 1 (1210)")
    print("Generated: Class 0 (+1000), Class 1 (+774)")
    print("Final dataset: Class 0 (2905), Class 1 (1984)")
