import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import logging
import numpy as np
import random
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Configure logging
log_file_path = r"UnSupervised_SimCLR_ResNet18_Finetuned_log.log"
model_save_path = "UnSupervised_SimCLR_ResNet18_Finetuned_model.pth"
output_csv='UnSupervised_SimCLR_ResNet18_evaluation_Finetuned_results.csv'
# Define the path to save the model
classifier_model_Path = "UnSupervised_SimCLR_ResNet18_evaluation_Finetuned_classifier_model.pth"

dataset_path = r'Rice_Leaf_Diease\TrainV3' 
simclr_epochs=10
classifier_epochs=10

# Set a seed for reproducibility
seed_value = 42  # You can choose any integer


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path, 'a')])


simclr_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224 for ResNet compatibility
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=(5, 9)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, return_label=False):
        self.dataset = dataset
        self.transform = transform
        self.return_label = return_label  # Add this flag

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]  # Get image and label
        
        # Apply transformations
        x_i = self.transform(img) if self.transform else img
        x_j = self.transform(img) if self.transform else img

        # Return two transformed images and label if specified
        if self.return_label:
            return x_i, x_j, target  # Used in extract_features
        else:
            return x_i, x_j  # Used in contrastive training (train_simclr)
  
# Define the SimCLR model with a projection head
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim):
        super(SimCLR, self).__init__()
        base_encoder.fc = nn.Identity()  # Remove the original fully connected layer
        self.base_encoder = base_encoder  # Set base encoder to ResNet backbone
        
        # Adjust the projection head for ResNet18, which outputs 512-dimensional features
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),  # Adjusted for ResNet18
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        # Forward pass through the base encoder
        features = self.base_encoder(x)  # Expected output shape: [batch_size, 512]
        
        # Flatten features if necessary
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        # Forward pass through the projection head
        projection = self.projection_head(features)
        logging.debug(f"Output from projection head shape: {projection.shape}")

        return projection

def nt_xent_loss(features, temperature=0.5):
    labels = torch.cat([torch.arange(features.shape[0] // 2) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    exp_logits = torch.exp(logits) * (1 - torch.eye(features.shape[0], device=features.device))
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_loss = -(labels * log_prob).sum(1).mean()
    return mean_loss

def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch_idx, (images_1, images_2, targets) in enumerate(loader):  # Expect targets as class indices
            images = images_1.to(device)  # Use one view for feature extraction

            # Extract features
            output = model.base_encoder(images)
            if output.dim() > 2:
                output = output.view(output.size(0), -1)

            features.append(output.cpu())
            labels.extend(targets.numpy())  # Append the integer labels

    features = torch.cat(features) if features else torch.empty(0)
    labels = torch.tensor(labels) if labels else torch.empty(0)
    
    #logging.info(f"Extracted features shape: {features.shape}, Labels shape: {labels.shape}")
    return features, labels


def extract_features_Old(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    features = []
    labels = []

    logging.info("Starting feature extraction...")  # Log the start of the process
    with torch.no_grad():  # Disable gradient tracking
        for batch_idx, (images, targets) in enumerate(loader):
            try:
                # Log the batch index and the shape of images
                logging.info(f"Processing batch {batch_idx + 1}/{len(loader)}, Image batch shape: {images.shape}")
                
                images = images.to(device)  # Move images to the specified device

                # Extract features using the base encoder, without the projection head
                output = model.base_encoder(images)

                # Log the shape of the output from the base encoder
                logging.debug(f"Output shape from base encoder: {output.shape}")

                # Flatten if necessary
                if output.dim() > 2:
                    output = output.view(output.size(0), -1)
                    logging.debug(f"Flattened output shape: {output.shape}")

                features.append(output.cpu())  # Store features on CPU
                labels.extend(targets.cpu().numpy())  # Convert targets to numpy and store labels on CPU

                logging.debug(f"Features collected so far: {[f.shape for f in features]}")
                logging.debug(f"Labels collected so far: {labels}")

            except Exception as e:
                logging.error(f"Error processing batch {batch_idx + 1}: {e}")  # Log any error that occurs in processing
                continue  # Skip to the next batch if there is an error

    try:
        # Concatenate features and convert labels to tensor
        if features:  # Check if features is not empty
            features = torch.cat(features)
        else:
            features = torch.empty(0)  # Ensure features is an empty tensor

        if labels:  # Check if labels is not empty
            labels = torch.tensor(labels)
        else:
            labels = torch.empty(0)  # Ensure labels is an empty tensor

        # Log shapes of features and labels
        logging.info(f"Extracted features shape: {features.shape}")
        logging.info(f"Labels shape: {labels.shape}")
        logging.info("Feature extraction completed successfully.")

    except Exception as e:
        logging.error(f"Error during feature concatenation: {e}")  # Log error if concatenation fails

    return features, labels

 # Fine-tuning classifier on extracted features
class Classifier(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(Classifier, self).__init__()
                self.fc = nn.Linear(input_dim, num_classes)
            
            def forward(self, x):
                return self.fc(x)

def train_simclr(model, dataloader, optimizer, initial_epochs, device):
    model.train()
    for epoch in range(initial_epochs):
        total_loss = 0.0
        logging.info(f"Starting epoch {epoch + 1}/{initial_epochs}")

        try:
            for x_i, x_j in dataloader:
                x_i, x_j = x_i.to(device), x_j.to(device)

                # Forward passes for both augmented views
                h_i, h_j = model(x_i), model(x_j)

                # Concatenate the representations
                features = torch.cat([h_i, h_j], dim=0)

                # Compute NT-Xent loss
                loss = nt_xent_loss(features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                #logging.info(f"Batch loss: {loss.item()}")

            avg_loss = total_loss / len(dataloader)
            logging.info(f"Epoch [{epoch + 1}/{initial_epochs}] completed, Average Loss: {avg_loss:.4f}")

        except Exception as e:
            logging.error(f"An error occurred during epoch {epoch + 1}: {e}")
            break  # Exit if there's an error

    # Save the model's state_dict to the specified path after all epochs are completed
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved at: {model_save_path}")
        
#  Define your functions for fine-tuning and evaluating the classifier
def fine_tune_classifier(classifier, features, labels, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    logging.info("Fine-tuning classifier")
    
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        
        features_device = features.to(device)
        labels_device = labels.to(device)

        if len(labels_device.shape) != 1:
            logging.warning("Labels are not 1D; attempting to flatten.")
            labels_device = labels_device.view(-1)
        
        #logging.info(f"Features shape: {features_device.shape}, Labels shape: {labels_device.shape}")
        
        outputs = classifier(features_device)
        loss = criterion(outputs, labels_device)
        loss.backward()
        optimizer.step()
        
        logging.info(f"Fine-tuning Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

def evaluate_classifier(classifier, features, labels, device, num_classes):
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(features.to(device))

        # Create a range for topk from 1 to num_classes
        topk = range(1, num_classes + 1)
        maxk = max(topk)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()

        if len(labels.shape) != 1:
            logging.error(f"Labels shape for accuracy calculation: {labels.shape}")
            raise ValueError("Labels should be a 1D tensor of class indices.")

        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            accuracies.append((correct_k / labels.size(0)).item())

        # Log and return the accuracies
        for k, acc in zip(topk, accuracies):
            logging.info(f"Top-{k} Accuracy: {acc * 100:.2f}%")

        # Calculate per-class accuracy
        pred_labels = pred[0]  # Only take the top-1 predictions for per-class metrics
        per_class_accuracy = []
        for class_idx in range(num_classes):
            # Filter out predictions and labels for the current class
            true_positives = ((pred_labels == class_idx) & (labels == class_idx)).sum().item()
            total_class_samples = (labels == class_idx).sum().item()
            accuracy_class = true_positives / total_class_samples if total_class_samples > 0 else 0
            per_class_accuracy.append(accuracy_class * 100)  # Convert to percentage
            logging.info(f"Class {class_idx} Accuracy: {accuracy_class * 100:.2f}%")

        # Calculate other metrics (precision, recall, F1-score) per class
        precision, recall, f1_score, _ = precision_recall_fscore_support(labels.cpu(), pred_labels.cpu(), labels=range(num_classes), zero_division=0)

        # Prepare data for CSV export
        results_df = pd.DataFrame({
            'Top-k': list(topk),
            'Accuracy': [acc * 100 for acc in accuracies]  # Convert to percentage
        })
        per_class_df = pd.DataFrame({
            'Class': list(range(num_classes)),
            'Per-Class Accuracy': per_class_accuracy,
            'Precision': precision * 100,  # Convert to percentage
            'Recall': recall * 100,        # Convert to percentage
            'F1-Score': f1_score * 100     # Convert to percentage
        })

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

        # Export results to CSV
        results_df.to_csv(output_csv, index=False)
        per_class_df.to_csv(output_csv.replace('.csv', '_per_class.csv'), index=False)
        
        logging.info(f"Evaluation results exported to {output_csv} and per-class results to {output_csv.replace('.csv', '_per_class.csv')}")

        return accuracies, per_class_accuracy, (precision, recall, f1_score)


# Main execution
if __name__ == "__main__":
    try:        
        # Set seeds for reproducibility
        seed_value = 42
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        random.seed(seed_value)

        logging.info("Started Training SimCLR")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Load dataset
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use standard normalization for ResNet
        ])

        # Apply SimCLR transformations directly in the dataset for contrastive learning
        dataset = ImageFolder(dataset_path)
        logging.info(f"Dataset path: {dataset_path}")

        simclr_dataset = SimCLRDataset(dataset, transform=simclr_transform, return_label=False)
        simclr_loader = DataLoader(simclr_dataset, batch_size=64, shuffle=True, num_workers=4)

        # Initialize SimCLR model with ResNet50 base encoder
        base_encoder = models.resnet18(pretrained=True)
        model = SimCLR(base_encoder, projection_dim=128).to(device)
        logging.info("SimCLR model initialized.")

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        logging.info("SimCLR training started.")
        # Run training using `simclr_loader` with correct transformations

        initial_epochs= simclr_epochs

        train_simclr(model, simclr_loader, optimizer,initial_epochs, device=device)
        logging.info("SimCLR training complete.")

        logging.info("Extracting features using the trained SimCLR model.")
        simclr_dataset_with_labels = SimCLRDataset(dataset, transform=simclr_transform, return_label=True)
        feature_loader = DataLoader(simclr_dataset_with_labels, batch_size=64, shuffle=False, num_workers=4)

        # Then call extract_features with this feature_loader
        features, labels = extract_features(model, feature_loader, device)

        logging.info("Feature extraction completed")    
        num_classes = len(dataset.classes)

        # Create a subset of the dataset (for fine-tuning)
        subset_fraction = 0.2  # Use 20% of the labeled dataset
        subset_size = int(len(features) * subset_fraction)
        subset_indices = random.sample(range(len(features)), subset_size)  # Randomly select subset_indices

        # Subset features and labels
        subset_features = features[subset_indices]
        subset_labels = labels[subset_indices]

        # Now, use these subset features and labels to fine-tune the classifier
        classifier = Classifier(input_dim=features.shape[1], num_classes=num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

        logging.info("Fine-tune classifier")
        # Fine-tune classifier using the subset of data
        fine_tune_classifier(classifier, subset_features, subset_labels, device, classifier_epochs)

        logging.info("Evaluate the classifier")
        # Evaluate the classifier on the subset
        evaluate_classifier(classifier, subset_features, subset_labels, device, num_classes)

        torch.save(classifier.state_dict(), classifier_model_Path)
        logging.info(f"Classifier model saved to {classifier_model_Path}")

    
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
