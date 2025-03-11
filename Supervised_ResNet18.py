import torch 
import torch.nn as nn
import torch.optim as optim
import logging
import random
import numpy as np
import csv
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report

# Configure logging
log_file_path = r"Supervised_ResNet18_training_log.log"
#dataset_path = r'Rice_Leaf_Diease\TrainV3' 
dataset_path = r'C:\Users\Tanee\OneDrive\Desktop\C anew\LUC\Rice_Leaf_Disease'
export_path="Supervised_ResNet18_evaluation_metrics.csv"
classwise_export_path="Supervised_ResNet18_classwise_performance.csv"
model_save_path = "supervised_resnet18_model.pth"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path, 'a')])

# Set seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Define data transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training loop
def train_supervised(model, dataloader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct_preds = 0
        total_samples = 0

        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = correct_preds / total_samples
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")

def evaluate(model, dataloader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    # Collect predictions and labels without calculating gradients
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Log overall metrics
    logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Precision: {precision * 100:.2f}%")
    logging.info(f"Recall: {recall * 100:.2f}%")
    logging.info("Confusion Matrix:")
    logging.info(f"\n{conf_matrix}")

    # Export overall metrics to a CSV file
    with open(export_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", accuracy])
        writer.writerow(["F1 Score", f1])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])
        writer.writerow(["Confusion Matrix", ""])
        for row in conf_matrix:
            writer.writerow([""] + row.tolist())

    # Generate class-wise metrics
    class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Export class-wise metrics to a separate CSV file
    with open(classwise_export_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Precision", "Recall", "F1 Score", "Support"])
        
        for class_name, metrics in class_report.items():
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                writer.writerow([class_name, metrics["precision"], metrics["recall"], metrics["f1-score"], metrics["support"]])

    logging.info(f"Class-wise metrics exported to {classwise_export_path}")

    return accuracy, f1, precision, recall, conf_matrix

# Main execution
if __name__ == "__main__":
    
    try:
        # Load the dataset and create an 80-20 train-test split
        dataset = ImageFolder(dataset_path, transform=transform)
        class_names = dataset.classes  # Save the class names
        logging.info(f"Loaded dataset from: {dataset_path}")

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

        # Initialize ResNet18 model for supervised learning
        logging.info("Initialize ResNet18 model for supervised learning")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = models.resnet18(pretrained=True)
        num_classes = len(dataset.classes)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)  # Replace the classification head
        model = base_model.to(device)

        # Define loss function and optimizer
        logging.info("Define loss function and optimizer")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        
        logging.info("Starting supervised training with ResNet-18")
        train_supervised(model, train_loader, optimizer, criterion, num_epochs=1, device=device)
        logging.info("Training complete. Evaluating on test data.")
        
        # Evaluation
        logging.info("Evaluation")
        evaluate(model, test_loader, device, class_names)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
