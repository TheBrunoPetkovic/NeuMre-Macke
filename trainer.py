import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

def extract_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            lbls = lbls.to(device)
            features = model.features(images)  # Dohvaća embeddinge iz features bloka
            embeddings.append(features.view(features.size(0), -1).cpu().numpy())
            labels.append(lbls.cpu().numpy())
    return np.vstack(embeddings), np.hstack(labels)

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

def get_all_preds_and_probs(model, loader):
    all_preds = []  
    all_labels = [] 
    all_probs = [] 

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images) 
            probs = torch.softmax(outputs, dim=1)  
            _, preds = torch.max(outputs, 1)

            print("Logiti:", outputs)
            print("Vjerojatnosti (probs):", probs)
            print("Predikcije (preds):", preds)
            print("Stvarne oznake (labels):", labels)
   
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

   
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    return all_labels, all_preds, all_probs

epoch_losses = []
epoch_accuracies = []
test_losses = []  
test_accuracies = []

if __name__ == "__main__":

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
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_acc)
            # Dodano: Evaluacija na test skupu
            model.eval()  # Postavi model u eval mod
            test_running_loss = 0.0
            test_correct = 0
            test_total = 0

            with torch.no_grad():  # Isključi gradijente za evaluaciju
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    test_running_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()

            # Izračunaj prosječni test gubitak i točnost
            test_loss = test_running_loss / len(test_loader.dataset)
            test_acc = 100 * test_correct / test_total

            test_losses.append(test_loss)  # Dodano: spremanje test gubitka
            test_accuracies.append(test_acc)  # Dodano: spremanje test točnosti

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))  

    val_acc = evaluate(model, val_loader)
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Spremanje modela
    torch.save(model.state_dict(), 'cat_breed_classifier.pth')
    np.save('test_losses.npy', np.array(test_losses))
    np.save('test_accuracies.npy', np.array(test_accuracies))
    # Spremanje embeddinga i pripadajućih klasa
    embeddings, labels = extract_embeddings(model, test_loader)
    np.save('embeddings.npy', embeddings)
    np.save('labels.npy', labels)

    # Spremanje predikcija i vjerojatnosti
    all_labels, all_preds, all_probs = get_all_preds_and_probs(model, test_loader)
    np.save('all_labels.npy', all_labels)
    np.save('all_preds.npy', all_preds)
    np.save('all_probs.npy', all_probs)


    # Generiraj matricu zabune
    conf_mat = confusion_matrix(all_labels, all_preds)

    # Spremi matricu zabune
    np.save('confusion_matrix.npy', conf_mat)

    np.save('epoch_losses.npy', np.array(epoch_losses))
    np.save('epoch_accuracies.npy', np.array(epoch_accuracies))

    # Generiraj izvještaj
    report = classification_report(all_labels, all_preds, target_names=[
        "Bombay", "British Shorthair", "Egyptian Mau", "Maine Coon", "Russian Blue"
    ])

    # Spremi izvještaj u tekstualnu datoteku
    with open('classification_report.txt', 'w') as f:
        f.write(report)

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


