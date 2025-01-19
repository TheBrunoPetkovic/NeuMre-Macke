import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import cv2
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_labels = np.load('all_labels.npy')
all_preds = np.load('all_preds.npy')

print("Stvarne oznake (all_labels):", all_labels)
print("Predikcije (all_preds):", all_preds)
# Generiraj matricu zabune
conf_mat = confusion_matrix(all_labels, all_preds)

# Definiraj nazive pasmina
class_names = ["Bombay", "British Shorthair", "Egyptian Mau", "Maine Coon", "Russian Blue"]
for idx, name in enumerate(class_names):
    print(f"Klasa {idx}: {name}")
# Iscrtavanje heatmap-a
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Matrica zabune - Prikaz")
plt.xlabel("Predviđene klase")
plt.ylabel("Stvarne klase")
plt.show()

# Učitaj embeddinge i pripadajuće oznake
embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')
##############################################Prikaz prostora značajki
# Pokreni t-SNE za smanjenje dimenzionalnosti na 2D
print("Pokrećem t-SNE, ovo može potrajati nekoliko minuta...")
tsne = TSNE(n_components=2, random_state=42, perplexity=20, n_iter=1500)

reduced_embeddings = tsne.fit_transform(embeddings)
print(f"Jedinstvene vrijednosti u labels: {np.unique(labels)}")

# Ispravno mapiranje boja na klase i ručna legenda
unique_labels = np.unique(labels)
colors = ['blue', 'orange', 'green', 'red', 'purple']  # Boje za 5 pasmina
class_names = ['Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Russian Blue']  # Imena pasmina

# Crtanje t-SNE rezultata
plt.figure(figsize=(10, 8))
for label, color, name in zip(unique_labels, colors, class_names):
    plt.scatter(
        reduced_embeddings[labels == label, 0],  # Filtriraj po oznakama
        reduced_embeddings[labels == label, 1],
        c=color, label=name, alpha=0.7
    )

# Dodavanje legende
plt.legend(loc='best', title='Pasmina mačaka')
plt.title('t-SNE vizualizacija prostora značajki')
plt.xlabel('Dimenzija 1')
plt.ylabel('Dimenzija 2')
plt.grid(True)
plt.show()


##################################################ROC i AUC

all_probs = np.load('all_probs.npy')  # Vjerojatnosti za svaku klasu
all_labels = np.load('all_labels.npy')  # Stvarne oznake

# Broj klasa
num_classes = all_probs.shape[1]

# Generiraj ROC krivulje i izračunaj AUC za svaku klasu
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])  # True vs predicted za klasu i
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Klasa {i} (AUC = {roc_auc:.2f})")

# Dodaj liniju za slučajni model
plt.plot([0, 1], [0, 1], 'k--', label='Slučajni model (AUC = 0.50)')

# Postavke grafa
plt.title("ROC krivulje za svaku klasu")
plt.xlabel("Lažno pozitivni rate (FPR)")
plt.ylabel("Istinito pozitivni rate (TPR)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

###############################################################

train_losses = np.load('epoch_losses.npy')
train_accuracies = np.load('epoch_accuracies.npy')
test_losses = np.load('test_losses.npy')
test_accuracies = np.load('test_accuracies.npy')

# Ispis duljine i nekoliko vrijednosti
print(f"Train Losses (length {len(train_losses)}):", train_losses[:5])
print(f"Train Accuracies (length {len(train_accuracies)}):", train_accuracies[:5])
print(f"Test Losses (length {len(test_losses)}):", test_losses[:5])
print(f"Test Accuracies (length {len(test_accuracies)}):", test_accuracies[:5])

# Kreiraj range za epohe
epochs = range(1, len(train_losses) + 1)

# Graf za gubitak
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Train loss", linestyle='-', marker='o')
plt.plot(epochs, test_losses, label="Test loss", linestyle='--', marker='x')
plt.title("Gubitak")
plt.xlabel("Epoha")
plt.ylabel("Gubitak")
plt.legend()
plt.grid()
plt.show()

# Graf za točnost
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label="Train loss", linestyle='-', marker='o')
plt.plot(epochs, test_accuracies, label="Test loss", linestyle='--', marker='x')
plt.title("Točnost")
plt.xlabel("Epoha")
plt.ylabel("Točnost (%)")
plt.legend()
plt.grid()
plt.show()

##############################################Grad CAM

# Funkcija za generiranje Grad-CAM vizualizacije
def generate_grad_cam(model, image, target_class, layer_name):
    model.eval()
    gradients = None
    activations = None

    # Hook za dohvaćanje gradijenata
    def save_gradients(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Hook za dohvaćanje aktivacija
    def save_activations(module, input, output):
        nonlocal activations
        activations = output

    # Pridruži hook na ciljni sloj
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_backward_hook(save_gradients)
            module.register_forward_hook(save_activations)

    # Forward pass
    outputs = model(image)
    pred_class = outputs.argmax(dim=1).item()  # Predviđena klasa
    print(f"Predviđena klasa: {pred_class}")

    # Backward pass za ciljnu klasu
    model.zero_grad()
    target = torch.zeros_like(outputs)
    target[0, target_class] = 1
    outputs.backward(gradient=target)

    # Računanje Grad-CAM mape
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    grad_cam = (weights * activations).sum(dim=1).squeeze(0).cpu().detach().numpy()
    print("Minimalna vrijednost Grad-CAM mape:", grad_cam.min())
    print("Maksimalna vrijednost Grad-CAM mape:", grad_cam.max())
    grad_cam = np.maximum(grad_cam, 0)  # ReLU
    
    return grad_cam

# Učitaj model
from trainer import CatBreedCNN  # Import modela iz trainer.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 5
model = CatBreedCNN(num_classes)
model.load_state_dict(torch.load('cat_breed_classifier.pth', map_location=device))
model.to(device)

# Učitaj sliku za analizu
image_path = "Macke/test/bombay/Bombay_205_jpg.rf.1cd36cc706f957624bb883d9394bff22.jpg" 
image = Image.open(image_path).convert('RGB')

# Transformacija slike
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_image = transform(image).unsqueeze(0).to(device)

# Generiraj Grad-CAM za ciljnu klasu
target_class = 0  # Promijeni na željenu klasu
layer_name = 'features.27'  
grad_cam = generate_grad_cam(model, input_image, target_class, layer_name)

if grad_cam.max() == grad_cam.min():
        print(f"Grad-CAM mapa ima konstantne vrijednosti: {grad_cam.max()}")
        grad_cam = np.zeros_like(grad_cam)  
else:
        grad_cam = grad_cam - grad_cam.min()  # Pomakni vrijednosti na 0+
        grad_cam = grad_cam / grad_cam.max()  # Normaliziraj na [0, 1]

# Prikaz slike s Grad-CAM-om
grad_cam_resized = cv2.resize(grad_cam, (image.size[0], image.size[1]))
heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_resized), cv2.COLORMAP_JET)
superimposed_image = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Grad-CAM vizualizacija")
plt.imshow(superimposed_image)
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Intenzitet važnosti', rotation=270, labelpad=20)

plt.show()
