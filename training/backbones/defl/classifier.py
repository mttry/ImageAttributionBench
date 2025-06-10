import torch
import torch.nn.functional as F

class ReferenceBasedClassifier:
    def __init__(self, reference_fingerprints, theta=3.5):
        self.reference_fingerprints = reference_fingerprints
        self.theta = theta
        # self.centers = self.calculate_centers()

    def calculate_centers(self): 
        centers = {}
        gan_fingerprints = []
        dm_fingerprints = []
        
        for label, fingerprints in self.reference_fingerprints.items():
            model_id = int(label.split('_')[-1]) 
            if model_id in {7, 8, 9, 10}:
                gan_fingerprints.append(fingerprints)
            elif model_id in {1, 2, 3, 4, 5, 6}:
                dm_fingerprints.append(fingerprints)
        
        if gan_fingerprints:
            centers['GAN'] = torch.mean(torch.cat(gan_fingerprints), dim=0).unsqueeze(0)  # 保持 (1, D) 维度
        else:
            print("Warning: No GAN fingerprints available to calculate center.")

        if dm_fingerprints:
            centers['DM'] = torch.mean(torch.cat(dm_fingerprints), dim=0).unsqueeze(0)  # 保持 (1, D) 维度
        else:
            print("Warning: No DM fingerprints available to calculate center.")
        
        self.centers = centers
        return centers

    def classify(self, test_fingerprint):

        batch_predictions = []
        for fingerprint in test_fingerprint:
            distances = {}
            
            for label, fingerprints in self.reference_fingerprints.items():
                dist = torch.cdist(fingerprint.unsqueeze(0), fingerprints.unsqueeze(0)).mean().item()
                distances[label] = dist

            if distances:
                min_label, min_distance = min(distances.items(), key=lambda x: x[1])
            else:
                batch_predictions.append("Unknown")
                continue

            batch_predictions.append(min_label)  
            # continue
            if min_distance <= self.theta:
                batch_predictions.append(min_label)  
            else:
                dm_models = {label: dist for label, dist in distances.items() if int(label.split('_')[-1]) in {1, 2, 3, 4, 5, 6}}
                gan_models = {label: dist for label, dist in distances.items() if int(label.split('_')[-1]) in {7, 8, 9, 10}}

                if 'GAN' in self.centers and 'DM' in self.centers:
                    dist_to_gan = F.pairwise_distance(fingerprint.unsqueeze(0), self.centers['GAN']).mean().item()
                    dist_to_dm = F.pairwise_distance(fingerprint.unsqueeze(0), self.centers['DM']).mean().item()
                    
                    if dist_to_gan < dist_to_dm:
                        if gan_models:
                            closest_model = min(gan_models, key=gan_models.get)
                        else:
                            closest_model = "Unknown"
                    else:
                        if dm_models:
                            closest_model = min(dm_models, key=dm_models.get)
                        else:
                            closest_model = "Unknown"
                    batch_predictions.append(closest_model)
                else:
                    batch_predictions.append("Unknown")

        return batch_predictions


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNNClassifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=11):
        super(SimpleNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def classify(self,x):
        return self.forward(x)

if __name__ == "__main__":
    input_features = torch.randn(32, 2048)  
    
    num_classes = 10  
    classifier = SimpleNNClassifier(input_dim=2048, num_classes=num_classes)
    
    outputs = classifier(input_features)
    print(outputs.shape) 
    
    labels = torch.randint(0, num_classes, (32,))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    print("Loss:", loss.item())
