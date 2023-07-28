import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset.targets = train_dataset.targets%2
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_dataset.targets = test_dataset.targets%2
# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the deep learning models
resnet_model = models.resnet50(pretrained=True)
alexnet_model = models.alexnet(pretrained=True)
inception_model = models.inception_v3(pretrained=True)
vgg16_model = models.vgg16(pretrained=True)

# Freeze the pre-trained model weights
for param in resnet_model.parameters():
    param.requires_grad = False

for param in alexnet_model.parameters():
    param.requires_grad = False

for param in inception_model.parameters():
    param.requires_grad = False

for param in vgg16_model.parameters():
    param.requires_grad = False

# Modify the last layer of each model for MNIST classification
num_classes = 2
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
alexnet_model.classifier[6] = nn.Linear(alexnet_model.classifier[6].in_features, num_classes)
inception_model.fc = nn.Linear(inception_model.fc.in_features, num_classes)
vgg16_model.classifier[6] = nn.Linear(vgg16_model.classifier[6].in_features, num_classes)

# Define the SVM model
svm = SVC()

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)
alexnet_model.to(device)
inception_model.to(device)
vgg16_model.to(device)

# Train and evaluate the models
def train(model, train_loader, model_name):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for images, labels in train_loader:
        # print(images.shape)
        if model_name == "ResNet" or model_name=="VGG16":
            images = images.reshape((-1, 1,28,28))
            # print(images.shape)
            images = np.repeat(images,3, axis=1)
        # print(images.shape)
        
        images = images.to(device)
        labels = labels.to(device)
        

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        print(f"loss: {loss}\n")
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader, model_name):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            # images = np.repeat(images,3, axis=1) #make three channels
            if model_name == "ResNet" or model_name == "VGG16":
                images = images.reshape((-1, 1,28,28))

                images = np.repeat(images,3, axis=1)
        
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return predictions, targets

# Training and evaluation loop for deep learning models
def evaluate_deep_learning_model(model, model_name):
    print(f"Now training {model_name}")
    print(model)
    train(model, train_loader, model_name)
    predictions, targets = evaluate(model, test_loader, model_name)

    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='macro')
    recall = recall_score(targets, predictions, average='macro')

    print(f"Metrics for {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print()

# Train and evaluate the deep learning models
# evaluate_deep_learning_model(resnet_model, "ResNet")
# evaluate_deep_learning_model(alexnet_model, "AlexNet")
# evaluate_deep_learning_model(inception_model, "Inception")
evaluate_deep_learning_model(vgg16_model, "VGG16")

# Prepare the data for SVM
scaler = StandardScaler()
x_train = train_dataset.data.reshape((-1, 28*28)).numpy()
x_test = test_dataset.data.reshape((-1, 28*28)).numpy()
y_train = train_dataset.targets.numpy()
y_test = test_dataset.targets.numpy()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train and evaluate the SVM
svm.fit(x_train, y_train)
svm_predictions = svm.predict(x_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='macro')
svm_recall = recall_score(y_test, svm_predictions, average='macro')

print("Metrics for SVM")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
