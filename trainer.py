import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #možda promjenit
])

train_dataset = datasets.ImageFolder(root='Macke/train', transform=transform)
val_dataset = datasets.ImageFolder(root='Macke/valid', transform=transform)
test_dataset = datasets.ImageFolder(root='Macke/test', transform=transform)

#velicina batcha
batch_size = 90

#koliko vrsta (pasmina?) mačaka
num_classes = 5

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class CatBreedCNN(nn.Module):
    def __init__(self, num_classes):
        super(CatBreedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # reduces size to 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # reduces size to 32x32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # reduces size to 16x16

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # reduces size to 8x8
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 8 * 8, 1024),  # Corrected input feature size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


#params
learning_rate = 0.001

#Ak je spremljen model preko ovoga se učita
#model = CatBreedCNN()
#model.load_state_dict(torch.load('cat_breed_classifier.pth'))
#model.to(device)

#a ova linija odma ispod se mora zakomentirat
model = CatBreedCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9) #moglo bi se uredit
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))  

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100 * correct / total

val_acc = evaluate(model, val_loader)
print(f"Validation Accuracy: {val_acc:.2f}%")

# Spremanje modela
torch.save(model.state_dict(), 'cat_breed_classifier.pth')

#-------------------------------------
# sve dolje je confusion matrix, radilo je sa v1 arhitekture mreze, nije prilagodeno za v2 i v3
#def get_all_preds(model, loader):
#    all_preds = torch.tensor([])
#    all_labels = torch.tensor([])
#    with torch.no_grad():
#        for images, labels in loader:
#            images = images.to(device)
#            outputs = model(images)
#            _, preds = torch.max(outputs, 1)
#            all_preds = torch.cat((all_preds, preds.cpu()), dim=0)
#            all_labels = torch.cat((all_labels, labels.cpu()), dim=0)
#   return all_labels.numpy(), all_preds.numpy()

#model = CatBreedCNN(num_classes).to(device)
#model.load_state_dict(torch.load('cat_breed_classifier.pth'))
#model.eval()

#labels, preds = get_all_preds(model, val_loader)

#conf_mat = confusion_matrix(labels, preds)

# Prikaz matrice zabune
#fig, ax = plt.subplots(figsize=(8,8))
#sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
#ax.set_xlabel('Predicted Labels')
#ax.set_ylabel('True Labels')
#ax.set_title('Confusion Matrix')
#ax.xaxis.set_ticklabels(['Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Russian Blue'])
#ax.yaxis.set_ticklabels(['Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Russian Blue'])
#plt.show()


