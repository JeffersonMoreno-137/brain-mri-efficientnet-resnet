# Clasificación de MRI con CNN — RESNET18
**Curso:** Técnicas de Deep Learning — MAIA  
**Dataset:** Brain Tumor MRI (7,023 imágenes, 4 clases)  
**Modelo:** RESNET18 con Transfer Learning

## 1. Imports y Configuración Global


```python
# ── Librerías estándar ──────────────────────────────────────────────────────
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from pathlib import Path

# ── PyTorch ─────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ── Métricas ─────────────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix, classification_report

print(f"PyTorch: {torch.__version__}")

# ── Configuración global ─────────────────────────────────────────────────────
DATASET_ROOT = Path("./dataset")  # <-- ajustar si es necesario
BATCH_SIZE   = 32
IMG_SIZE     = 224
EPOCHS       = 10
LR           = 1e-3
SEED         = 42
NUM_CLASES   = 4
DEVICE = torch.device("mps")

torch.manual_seed(SEED)
print(f"Dispositivo detectado: {DEVICE}")
```

    PyTorch: 2.8.0
    Dispositivo detectado: mps


## 2. Exploración del Dataset


```python
CLASES_NOMBRES = ["glioma", "healthy", "meningioma", "pituitary"]

# Conteo por clase
print(f"{'CLASE':<15} {'IMÁGENES':>8}")
print("=" * 25)
total = 0
for clase in CLASES_NOMBRES:
    carpeta = DATASET_ROOT / clase
    n = len(list(carpeta.glob("*.jpg"))) + len(list(carpeta.glob("*.png")))
    total += n
    print(f"{clase:<15} {n:>8}")
print("=" * 25)
print(f"{'TOTAL':<15} {total:>8}")

# Visualización de muestras
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("Fig. 1 — Muestra representativa por clase", fontsize=12, fontweight="bold")

for col, clase in enumerate(CLASES_NOMBRES):
    carpeta = DATASET_ROOT / clase
    imagenes = list(carpeta.glob("*.jpg")) + list(carpeta.glob("*.png"))
    img = mpimg.imread(random.choice(imagenes))
    axes[col].imshow(img, cmap="gray")
    axes[col].set_title(f"{clase}\n({len(imagenes)} imgs)", fontsize=10)
    axes[col].axis("off")

plt.tight_layout()
plt.savefig("fig1_muestras.png", dpi=150, bbox_inches="tight")
plt.show()
```

    CLASE           IMÁGENES
    =========================
    glioma              1621
    healthy             2000
    meningioma          1645
    pituitary           1757
    =========================
    TOTAL               7023



    
![png](output_4_1.png)
    


## 3. Pipeline de Datos


```python
# Transformaciones para entrenamiento (con augmentations)
# Referencia pesos de RESNET18
#    https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),   # EfficientNet espera 3 canales
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Transformaciones para prueba (sin augmentations)
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Cargar dataset completo
dataset_completo = datasets.ImageFolder(root=DATASET_ROOT, transform=transform_train)
CLASES = dataset_completo.classes
print("Clases detectadas:", CLASES)
print("Mapeo:", dataset_completo.class_to_idx)

# División 80/20
n_train = int(0.8 * len(dataset_completo))
n_test  = len(dataset_completo) - n_train
train_set, test_set = random_split(dataset_completo, [n_train, n_test])
test_set.dataset.transform = transform_test

# DataLoaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

print(f"\nEntrenamiento: {n_train} imágenes ({len(train_loader)} batches)")
print(f"Prueba:        {n_test} imágenes ({len(test_loader)} batches)")
```

    Clases detectadas: ['glioma', 'healthy', 'meningioma', 'pituitary']
    Mapeo: {'glioma': 0, 'healthy': 1, 'meningioma': 2, 'pituitary': 3}
    
    Entrenamiento: 5618 imágenes (176 batches)
    Prueba:        1405 imágenes (44 batches)


## 4. Modelo — RESNET18 con Transfer Learning


```python
def construir_modelo(num_clases, device):
    """Carga RESNET18 pre-entrenado y reemplaza el clasificador final."""
    modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Congelar todas las capas base
    for param in modelo.parameters():
        param.requires_grad = False
    
    # Reemplazar clasificador: 1280 → NUM_CLASES
    in_features = modelo.fc.in_features
    modelo.fc = nn.Linear(in_features, num_clases)
    return modelo.to(device)

modelo = construir_modelo(NUM_CLASES, DEVICE)

total_params     = sum(p.numel() for p in modelo.parameters())
trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

print(f"Parámetros totales:     {total_params:,}")
print(f"Parámetros entrenables: {trainable_params:,}")
print(f"Parámetros congelados:  {total_params - trainable_params:,}")
```

    Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /Users/jemoreno/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth


    100%|██████████████████████████████████████| 44.7M/44.7M [00:01<00:00, 33.4MB/s]


    Parámetros totales:     11,178,564
    Parámetros entrenables: 2,052
    Parámetros congelados:  11,176,512


## 5. Entrenamiento


```python
def train_epoch(modelo, loader, criterion, optimizer, device):
    modelo.train()
    loss_total, correctos, total = 0, 0, 0
    for imagenes, etiquetas in loader:
        imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
        optimizer.zero_grad()
        salidas = modelo(imagenes)
        loss = criterion(salidas, etiquetas)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        _, preds = torch.max(salidas, 1)
        correctos += (preds == etiquetas).sum().item()
        total += etiquetas.size(0)
    return loss_total / len(loader), correctos / total


def eval_epoch(modelo, loader, criterion, device):
    modelo.eval()
    loss_total, correctos, total = 0, 0, 0
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            salidas = modelo(imagenes)
            loss = criterion(salidas, etiquetas)
            loss_total += loss.item()
            _, preds = torch.max(salidas, 1)
            correctos += (preds == etiquetas).sum().item()
            total += etiquetas.size(0)
    return loss_total / len(loader), correctos / total


# Configurar loss y optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelo.fc.parameters(), lr=LR)

# Loop de entrenamiento
historial = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
mejor_acc = 0.0

print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12} {'Tiempo'}")
print("=" * 68)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_epoch(modelo, train_loader, criterion, optimizer, DEVICE)
    test_loss,  test_acc  = eval_epoch(modelo, test_loader,  criterion, DEVICE)
    elapsed = time.time() - t0

    historial["train_loss"].append(train_loss)
    historial["train_acc"].append(train_acc)
    historial["test_loss"].append(test_loss)
    historial["test_acc"].append(test_acc)

    marca = ""
    if test_acc > mejor_acc:
        mejor_acc = test_acc
        torch.save(modelo.state_dict(), "mejor_modelo.pth")
        marca = " ← mejor"

    print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.4f} {test_loss:<12.4f} {test_acc:<12.4f} {elapsed:.1f}s{marca}")

print(f"\nMejor Test Accuracy: {mejor_acc:.4f}")
```

    Epoch    Train Loss   Train Acc    Test Loss    Test Acc     Tiempo
    ====================================================================
    1        0.6730       0.7558       0.4478       0.8448       110.1s ← mejor
    2        0.4109       0.8549       0.3653       0.8762       105.8s ← mejor
    3        0.3519       0.8738       0.3394       0.8819       105.5s ← mejor
    4        0.3219       0.8806       0.3042       0.8925       105.7s ← mejor
    5        0.2999       0.8918       0.3143       0.8875       106.0s
    6        0.2977       0.8904       0.2864       0.8932       105.9s ← mejor
    7        0.2775       0.8996       0.2693       0.8982       106.8s ← mejor
    8        0.2689       0.8966       0.2723       0.9011       105.9s ← mejor
    9        0.2596       0.9067       0.2635       0.9032       105.6s ← mejor
    10       0.2562       0.9089       0.2637       0.9060       105.5s ← mejor
    
    Mejor Test Accuracy: 0.9060


## 6. Curvas de Aprendizaje


```python
epochs_range = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Fig. 2 — Curvas de aprendizaje — EfficientNetB0", fontsize=12, fontweight="bold")

ax1.plot(epochs_range, historial["train_loss"], label="Train", marker="o")
ax1.plot(epochs_range, historial["test_loss"],  label="Test",  marker="o")
ax1.set_title("Loss por epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs_range, historial["train_acc"], label="Train", marker="o")
ax2.plot(epochs_range, historial["test_acc"],  label="Test",  marker="o")
ax2.set_title("Accuracy por epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig2_curvas.png", dpi=150, bbox_inches="tight")
plt.show()
```


    
![png](output_12_0.png)
    


## 7. Evaluación Cuantitativa


```python
# Cargar mejor modelo
modelo.load_state_dict(torch.load("mejor_modelo.pth", map_location=DEVICE))
modelo.eval()

# Obtener todas las predicciones
todas_etiquetas, todas_predicciones = [], []
with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        salidas = modelo(imagenes.to(DEVICE))
        _, preds = torch.max(salidas, 1)
        todas_etiquetas.extend(etiquetas.numpy())
        todas_predicciones.extend(preds.cpu().numpy())

# Reporte por clase
print("— Reporte de clasificación —")
print(classification_report(todas_etiquetas, todas_predicciones,
                            target_names=CLASES, digits=4))

# Matriz de confusión
cm = confusion_matrix(todas_etiquetas, todas_predicciones)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im)
ax.set_xticks(range(NUM_CLASES))
ax.set_yticks(range(NUM_CLASES))
ax.set_xticklabels(CLASES, rotation=45, ha="right")
ax.set_yticklabels(CLASES)
ax.set_xlabel("Predicción", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
ax.set_title("Fig. 3 — Matriz de Confusión — EfficientNetB0", fontsize=12, fontweight="bold")

for i in range(NUM_CLASES):
    for j in range(NUM_CLASES):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=13, color=color, fontweight="bold")

plt.tight_layout()
plt.savefig("fig3_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
```

    — Reporte de clasificación —
                  precision    recall  f1-score   support
    
          glioma     0.9164    0.8943    0.9052       331
         healthy     0.9603    0.9627    0.9615       402
      meningioma     0.8710    0.7594    0.8114       320
       pituitary     0.8675    0.9858    0.9229       352
    
        accuracy                         0.9060      1405
       macro avg     0.9038    0.9005    0.9002      1405
    weighted avg     0.9064    0.9060    0.9044      1405
    



    
![png](output_14_1.png)
    


## 8. Resultados Cualitativos — Aciertos y Fallos


```python
def desnormalizar(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    img  = img * std + mean
    return np.clip(img[:, :, 0], 0, 1)

# Recolectar 4 aciertos y 4 fallos
aciertos, fallos = [], []
modelo.eval()

with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        salidas  = modelo(imagenes.to(DEVICE))
        _, preds = torch.max(salidas, 1)
        for i in range(len(etiquetas)):
            real, pred = etiquetas[i].item(), preds[i].item()
            if pred == real and len(aciertos) < 4:
                aciertos.append((imagenes[i], real, pred))
            elif pred != real and len(fallos) < 4:
                fallos.append((imagenes[i], real, pred))
        if len(aciertos) == 4 and len(fallos) == 4:
            break

# Figura
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Fig. 4 — Predicciones cualtitativas — EfficientNetB0",
             fontsize=12, fontweight="bold")

for col, (img, real, pred) in enumerate(aciertos):
    axes[0, col].imshow(desnormalizar(img), cmap="gray")
    axes[0, col].set_title(f"Real: {CLASES[real]}\nPred: {CLASES[pred]}",
                           fontsize=9, color="green", fontweight="bold")
    axes[0, col].axis("off")

for col, (img, real, pred) in enumerate(fallos):
    axes[1, col].imshow(desnormalizar(img), cmap="gray")
    axes[1, col].set_title(f"Real: {CLASES[real]}\nPred: {CLASES[pred]}",
                           fontsize=9, color="red", fontweight="bold")
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Aciertos ✓", fontsize=11, fontweight="bold", color="green")
axes[1, 0].set_ylabel("Fallos ✗",   fontsize=11, fontweight="bold", color="red")

plt.tight_layout()
plt.savefig("fig4_cualitativos.png", dpi=150, bbox_inches="tight")
plt.show()
```


    
![png](output_16_0.png)
    


---
---

## 9. FINE-TUNING DEL MODELO RESNET18



```python
# Se descongela el bloque 'layer4' y la capa 'fc'
for name, child in modelo.named_children():
    if name in ['layer4', 'fc']:
        for param in child.parameters():
            param.requires_grad = True
    else:
        for param in child.parameters():
            param.requires_grad = False

trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f"Nuevos parámetros entrenables: {trainable_params:,}")

```

    Nuevos parámetros entrenables: 8,395,780



```python
# Se propone un Learning Rate (LR) más bajo, mismo optimizador
LR_FINETUNE = 1e-4 
optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=LR_FINETUNE)

# El mismo criterio de pérdida se mantiene
criterion = nn.CrossEntropyLoss()
```


```python
EPOCHS_FT = 10
# historial_ft permitirá comparar las curvas después del fine-tuning
historial_ft = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12} {'Tiempo'}")
print("=" * 68)

for epoch in range(1, EPOCHS_FT + 1):
    t0 = time.time()
    # Se utilizan las mismas funciones train/eval pero con el nuevo optimizer_ft
    train_loss, train_acc = train_epoch(modelo, train_loader, criterion, optimizer_ft, DEVICE)
    test_loss,  test_acc  = eval_epoch(modelo, test_loader,  criterion, DEVICE)
    elapsed = time.time() - t0

    historial_ft["train_loss"].append(train_loss)
    historial_ft["train_acc"].append(train_acc)
    historial_ft["test_loss"].append(test_loss)
    historial_ft["test_acc"].append(test_acc)

    marca = ""
    # Se compara contra el mejor_acc anterior para ver si se supera el 90.6%
    if test_acc > mejor_acc:
        mejor_acc = test_acc
        torch.save(modelo.state_dict(), "mejor_modelo_ft.pth")
        marca = " ← mejor (FT)"

    print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.4f} {test_loss:<12.4f} {test_acc:<12.4f} {elapsed:.1f}s{marca}")

print(f"\nFinalizado. Mejor precisión total tras Fine-tuning: {mejor_acc:.4f}")
```

    Epoch    Train Loss   Train Acc    Test Loss    Test Acc     Tiempo
    ====================================================================
    1        0.2079       0.9254       0.1302       0.9488       109.4s ← mejor (FT)
    2        0.0387       0.9868       0.0847       0.9694       108.6s ← mejor (FT)
    3        0.0161       0.9945       0.0798       0.9751       107.8s ← mejor (FT)
    4        0.0060       0.9988       0.0907       0.9673       108.2s
    5        0.0086       0.9973       0.1444       0.9594       108.2s
    6        0.0303       0.9904       0.0868       0.9687       108.0s
    7        0.0323       0.9891       0.1145       0.9694       108.3s
    8        0.0189       0.9931       0.0794       0.9772       108.1s ← mejor (FT)
    9        0.0097       0.9964       0.0463       0.9829       108.5s ← mejor (FT)
    10       0.0051       0.9986       0.0571       0.9829       108.1s
    
    Finalizado. Mejor precisión total tras Fine-tuning: 0.9829


## 9.1 CURVAS DE APRENDIZAJE


```python
epochs_range = range(1, EPOCHS_FT + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("fig 5. Curvas de aprendizaje - ResNet18 + Fine-Tuning ", fontsize=12, fontweight="bold")

ax1.plot(epochs_range, historial_ft["train_loss"], label="Train", marker="o")
ax1.plot(epochs_range, historial_ft["test_loss"],  label="Test",  marker="o")
ax1.set_title("Loss por epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs_range, historial_ft["train_acc"], label="Train", marker="o")
ax2.plot(epochs_range, historial_ft["test_acc"],  label="Test",  marker="o")
ax2.set_title("Accuracy por epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig5_curvas_fine_tuning.png", dpi=150, bbox_inches="tight")
plt.show()
```


    
![png](output_23_0.png)
    



```python
### COMPARACIÓN VISUAL DE ANTES Y DESPUÉS DE FINE-TUNING
```


```python
# Se concatenan los historiales de la fase 1 (red congelada) y fase 2 (fine-tuning)
full_train_loss = historial["train_loss"] + historial_ft["train_loss"]
full_test_loss  = historial["test_loss"]  + historial_ft["test_loss"]
full_train_acc  = historial["train_acc"]  + historial_ft["train_acc"]
full_test_acc   = historial["test_acc"]   + historial_ft["test_acc"]

total_epochs = len(full_train_loss)
epochs_range = range(1, total_epochs + 1)

# Creación de la gráfica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 6. Comparación Transfer Learning + Fine-Tuning (ResNet18)", 
             fontsize=14, fontweight="bold")

# Gráfica 1 - Pérdida Total (en escala logarítmica)
ax1.plot(epochs_range, full_train_loss, label="Train Loss", color="royalblue", linewidth=2)
ax1.plot(epochs_range, full_test_loss, label="Test Loss", color="darkorange", linewidth=2)
ax1.axvline(x=10, color='red', linestyle='--', alpha=0.6, label='Inicio Fine-Tuning') # Línea vertical
ax1.set_title("Pérdida Total", fontsize=12)
ax1.set_xlabel("Épocas")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_yscale('log') # Escala logarítmica para apreciar diferencias pequeñas al final 
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Gráfica 2 de Accuracy
ax2.plot(epochs_range, full_train_acc, label="Train Acc", color="royalblue", linewidth=2)
ax2.plot(epochs_range, full_test_acc, label="Test Acc", color="darkorange", linewidth=2)
ax2.axvline(x=10, color='red', linestyle='--', alpha=0.6, label='Inicio Fine-Tuning')
ax2.set_title("Exactitud Total", fontsize=12)
ax2.set_xlabel("Épocas")
ax2.set_ylabel("Precisión (%)")
ax2.legend(loc='lower right')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("evidencia_final_resnet18.png", dpi=200, bbox_inches="tight")
plt.show()
```


    
![png](output_25_0.png)
    


### 10. EVALUACIÓN CUANTITATIVA - FINE-TUNING


```python
# Cargar mejor modelo
modelo.load_state_dict(torch.load("mejor_modelo_ft.pth", map_location=DEVICE))
modelo.eval()

# Obtener todas las predicciones
todas_etiquetas, todas_predicciones = [], []
with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        salidas = modelo(imagenes.to(DEVICE))
        _, preds = torch.max(salidas, 1)
        todas_etiquetas.extend(etiquetas.numpy())
        todas_predicciones.extend(preds.cpu().numpy())

# Reporte por clase
print("— Reporte de clasificación —")
print(classification_report(todas_etiquetas, todas_predicciones,
                            target_names=CLASES, digits=4))

# Matriz de confusión
cm = confusion_matrix(todas_etiquetas, todas_predicciones)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im)
ax.set_xticks(range(NUM_CLASES))
ax.set_yticks(range(NUM_CLASES))
ax.set_xticklabels(CLASES, rotation=45, ha="right")
ax.set_yticklabels(CLASES)
ax.set_xlabel("Predicción", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
ax.set_title("Fig. 7 — Matriz de Confusión — ResNet18 + Fine-Tuning", fontsize=12, fontweight="bold")

for i in range(NUM_CLASES):
    for j in range(NUM_CLASES):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=13, color=color, fontweight="bold")

plt.tight_layout()
plt.savefig("fig7_confusion_fine-tuning.png", dpi=150, bbox_inches="tight")
plt.show()
```

    — Reporte de clasificación —
                  precision    recall  f1-score   support
    
          glioma     0.9878    0.9758    0.9818       331
         healthy     0.9975    0.9925    0.9950       402
      meningioma     0.9485    0.9781    0.9631       320
       pituitary     0.9943    0.9830    0.9886       352
    
        accuracy                         0.9829      1405
       macro avg     0.9820    0.9824    0.9821      1405
    weighted avg     0.9832    0.9829    0.9830      1405
    



    
![png](output_27_1.png)
    



```python
# Definición manual de los valores (VP, VN, FP, FN)
datos = np.array([
    [323, 1070, 4, 8],  # glioma
    [399, 1002, 1, 3],  # healthy
    [313, 1068, 17, 7], # meningioma
    [346, 1051, 2, 6]   # pituitary
])

# Etiquetas para las filas y columnas
clases = ["glioma", "healthy", "meningioma", "pituitary"]
columnas = ["VP", "VN", "FP", "FN"]

# Crear la tabla
tabla_diagnostico = pd.DataFrame(datos, index=clases, columns=columnas)

# Mostrar la tabla sencilla
print("TABLA DE RESULTADOS (RESNET18 + FT)\n")
print(tabla_diagnostico)
```

    TABLA DE RESULTADOS (RESNET18 + FT)
    
                 VP    VN  FP  FN
    glioma      323  1070   4   8
    healthy     399  1002   1   3
    meningioma  313  1068  17   7
    pituitary   346  1051   2   6


## 11. Resultados Cualitativos — Aciertos y Fallos - Fine-Tuning


```python
# Recolectar 4 aciertos y 4 fallos
aciertos, fallos = [], []
modelo.load_state_dict(torch.load("mejor_modelo_ft.pth", map_location=DEVICE))
modelo.eval()

with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        salidas  = modelo(imagenes.to(DEVICE))
        _, preds = torch.max(salidas, 1)
        for i in range(len(etiquetas)):
            real, pred = etiquetas[i].item(), preds[i].item()
            if pred == real and len(aciertos) < 4:
                aciertos.append((imagenes[i], real, pred))
            elif pred != real and len(fallos) < 4:
                fallos.append((imagenes[i], real, pred))
        if len(aciertos) == 4 and len(fallos) == 4:
            break

# Figura
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Fig. 8 — Análisis Cualitativo Final: ResNet18 + Fine-Tuning",
             fontsize=12, fontweight="bold")
# Mostrar Aciertos
for col, (img, real, pred) in enumerate(aciertos):
    axes[0, col].imshow(desnormalizar(img), cmap="gray")
    axes[0, col].set_title(f"Real: {CLASES[real]}\nPred: {CLASES[pred]}",
                           fontsize=9, color="green", fontweight="bold")
    axes[0, col].axis("off")

# Mostrar Fallos
for col, (img, real, pred) in enumerate(fallos):
    axes[1, col].imshow(desnormalizar(img), cmap="gray")
    axes[1, col].set_title(f"Real: {CLASES[real]}\nPred: {CLASES[pred]}",
                           fontsize=9, color="red", fontweight="bold")
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Aciertos", fontsize=11, fontweight="bold", color="green")
axes[1, 0].set_ylabel("Fallos",   fontsize=11, fontweight="bold", color="red")

plt.tight_layout()
plt.savefig("fig8_cualitativos_final.png", dpi=150, bbox_inches="tight")
plt.show()
```


    
![png](output_30_0.png)
    



```python

```
