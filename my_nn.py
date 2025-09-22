import torch
import torch.nn as nn
import math
from nn_layer import scalarMul, ReLUFunc, SigmoidFunc, Matmul, TiledMatmulReLU
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class CustomNN(nn.Module):
    def __init__(self,in_features=28*28, hidden=256, out_features=10):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, hidden, device='cuda'))
        self.b1 = nn.Parameter(torch.zeros(hidden, device='cuda'))
        self.fc2 = nn.Linear(hidden, out_features).to(device='cuda')
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.zeros_(self.b1)
        
    def forward(self, x):
        device = self.W.device
        x= x.to(device)
        x = x.view(x.size(0), -1).contiguous()  # flatten
        x = TiledMatmulReLU.apply(x, self.W, self.b1)
        logits = self.fc2(x)
        return logits
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("This model requires a CUDA-capable GPU.")
    else:
        print("Using GPU:", torch.cuda.get_device_name(0))
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        num_sample = len(dataset)
        fold_size = num_sample // 5
        folds = [fold_size] * 5
        folds[-1] += num_sample - sum(folds)  # Adjust last fold to include any remainder

        subsets = random_split(dataset, folds, generator=torch.Generator().manual_seed(42))
        
        test_set = subsets[0]
        train_dataset = torch.utils.data.ConcatDataset(subsets[1:])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        
        model = CustomNN(in_features=28*28, hidden=256, out_features=10).to(device)
    
        #optimizer and loss
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)   
        criterion = nn.CrossEntropyLoss()
        epochs = 50
        
        def test(model, test_loader, criterion, device):
            model.eval()
            total_loss = 0.0
            correct = 0
            total_samples = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.view(images.size(0), -1).to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)

                    batch_size = images.size(0)
                    total_loss += loss.item() * batch_size
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total_samples += batch_size

            avg_loss = total_loss / total_samples
            accuracy = 100.0 * correct / total_samples
            print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n')
            return avg_loss, accuracy
        
        
        for epoch in range(epochs):
            model.train()
            for batch_idx,(images, labels) in enumerate(train_loader):
                input_data = images.view(images.size(0), -1).to(device) # flatten 28x28 -> 784
                labels = labels.to(device)  # Convert labels to float and add an extra dimension
                #forward pass
                logits = model(input_data)
                #computeloss
                loss = criterion(logits, labels)
                #backward pass
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    pred = logits.argmax(dim=1)
                    acc = (pred == labels).float().mean().item()
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {acc:.4f}")      

            test_loss, test_acc = test(model, test_loader, criterion, device)
            print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")