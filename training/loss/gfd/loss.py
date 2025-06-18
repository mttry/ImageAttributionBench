import torch  
import torch.nn as nn  
import torchvision.models as models  
import torch.nn.functional as F  

class AuxiliaryClassificationLoss(nn.Module):  
    def __init__(self):  
        super(AuxiliaryClassificationLoss, self).__init__()  
        self.criterion = nn.CrossEntropyLoss() 

    def forward(self, auxiliary_prediction, y):  
        loss = self.criterion(auxiliary_prediction, y)
        return loss
    

class AdversarialLoss(nn.Module):  
    def __init__(self):  
        super(AdversarialLoss, self).__init__()  

    def discriminator_loss(self, fake_pred, real_pred):  
        loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        return loss_fake + loss_real

    def generator_loss(self, fake_pred):
        loss_G = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        return loss_G

    def forward(self, real_predictions, fake_predictions):  
        loss_D = self.discriminator_loss(fake_predictions, real_predictions)  
        loss_G = self.generator_loss(fake_predictions)  

        return loss_D, loss_G  
    
class PerceptualLoss(nn.Module):  
    def __init__(self, device):  
        super(PerceptualLoss, self).__init__()  
        vgg = models.vgg16(pretrained=True).features  
        self.vgg = vgg.eval().to(device)  
        
        for param in self.vgg.parameters():  
            param.requires_grad = False  

    def forward(self, fingerprinted_image, real_image):  
        features_fingerprint = self.vgg(fingerprinted_image)  
        features_real = self.vgg(real_image)  

        loss = F.mse_loss(features_fingerprint, features_real) 

        return loss  
    
class LatentClassificationLoss(nn.Module):  
    def __init__(self):  
        super(LatentClassificationLoss, self).__init__()  
        self.criterion = nn.CrossEntropyLoss() 

    def forward(self, latent_code, labels):  
        # Lz_G: Latent classification loss  
        loss = self.criterion(latent_code, labels) 
        return loss     
    
omega1 = 1.0  
omega2 = 1.0  
omega3 = 1.0 
omega4 = 1.0  

class CombinedLoss(nn.Module):  
    def __init__(self,device):  
        super(CombinedLoss, self).__init__()  
        self.latent_loss_function = LatentClassificationLoss()  
        self.adversarial_loss_function = AdversarialLoss()  
        self.perceptual_loss_function = PerceptualLoss(device)  
        self.auxiliary_loss_function = AuxiliaryClassificationLoss()  

    def calculate_generator_loss(self, real_image, fingerprinted_image, real_labels, gen_labels,real_pred_prob,fake_pred_prob,D_real,D_fingerprint,auxiliary_prediction_xy):  
        labels = torch.cat((gen_labels, real_labels), dim=0)  
        pred_probs = torch.cat((fake_pred_prob,real_pred_prob), dim=0)  
        Lz_G = self.latent_loss_function(pred_probs, labels)  
        _, Ladv_G = self.adversarial_loss_function(D_fingerprint, D_real)  
        Lcls_G = self.auxiliary_loss_function(auxiliary_prediction_xy, gen_labels)  
        Lpercept_G = self.perceptual_loss_function(fingerprinted_image, real_image)  

        LG = (omega1 * Lz_G) + (omega2 * Ladv_G) + (omega3 * Lcls_G) + (omega4 * Lpercept_G)  
        loss_dict = {"Lz_G": Lz_G, "Ladv_G": Ladv_G, "Lcls_G": Lcls_G, "Lpercept_G": Lpercept_G, "overall": LG}
        return loss_dict  

    def calculate_discriminator_loss(self, D_real, D_fingerprint, real_image_logits, real_labels):  
        Ladv_D, _ = self.adversarial_loss_function(D_fingerprint, D_real) 
        Lcls_C = self.auxiliary_loss_function(real_image_logits, real_labels) 
        loss_dict = {"Lcls_C": Lcls_C, "Ladv_D": Ladv_D, "overall": Lcls_C + Ladv_D}
        return  loss_dict