import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class UNetEncoder(nn.Module):  
    def __init__(self, in_channels, base_channels):  
        super(UNetEncoder, self).__init__()  
        self.enc1 = self.conv_block(in_channels, base_channels)  # Level 1  
        self.enc2 = self.conv_block(base_channels, base_channels * 2)  # Level 2  
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)  # Level 3  
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)  # Level 4  

    def conv_block(self, in_channels, out_channels):  
        return nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
        )  

    def forward(self, x):  
        enc1 = self.enc1(x)  
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  
        enc3 = self.enc3(F.max_pool2d(enc2, 2))  
        enc4 = self.enc4(F.max_pool2d(enc3, 2))  
        return enc1, enc2, enc3, enc4  


class UNetDecoder(nn.Module):  
    def __init__(self, base_channels):  
        super(UNetDecoder, self).__init__()  
        self.dec4 = self.upconv_block(base_channels * 16, base_channels * 8)  # Level 4  
        self.dec3 = self.upconv_block(base_channels * 8 * 2, base_channels * 4)  # Level 3  
        self.dec2 = self.upconv_block(base_channels * 4 * 2, base_channels * 2)  # Level 2  
        self.dec1 = self.upconv_block(base_channels * 2 * 2, base_channels)  # Level 1  
        self.final_conv = nn.Conv2d(base_channels*2, 3, kernel_size=1) 

    def upconv_block(self, in_channels, out_channels):  
        return nn.Sequential(  
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
        )  

    def forward(self, enc1, enc2, enc3, enc4, bottleneck):  
        dec4 = self.dec4(bottleneck)  
        dec4 = torch.cat((dec4, enc4), dim=1) 
        dec3 = self.dec3(dec4)  
        dec3 = torch.cat((dec3, enc3), dim=1)   
        dec2 = self.dec2(dec3)  
        dec2 = torch.cat((dec2, enc2), dim=1) 
        dec1 = self.dec1(dec2)  
        dec1 = torch.cat((dec1, enc1), dim=1)   

        out = self.final_conv(dec1)    
        return out


class UNetGenerator(nn.Module):  
    def __init__(self, in_channels=3, base_channels=64,num_classes=11):  
        super(UNetGenerator, self).__init__()  
        self.encoder = UNetEncoder(in_channels, base_channels)  
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)  # Bottleneck  
        self.decoder = UNetDecoder(base_channels)  

        self.classification_head = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),   
            nn.Flatten(),                 
            nn.Linear(base_channels * 16, num_classes)   
        )  

    def conv_block(self, in_channels, out_channels):  
        return nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
        )  

    def forward(self, x):  
        enc1, enc2, enc3, enc4 = self.encoder(x)  
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))  
        out = self.decoder(enc1, enc2, enc3, enc4, bottleneck)  
        latent_code = bottleneck
        class_prediction = self.classification_head(latent_code)  
        return out, class_prediction    
    
    def get_classifer_logits(self,x):
        enc1, enc2, enc3, enc4 = self.encoder(x)  
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))  
        latent_code = bottleneck
        class_prediction = self.classification_head(latent_code)
        return class_prediction


if __name__ == "__main__":  
    generator = UNetGenerator()  
    input_tensor = torch.randn(1, 3, 256, 256)   
    output_tensor, class_prediction = generator(input_tensor)  
    print("Output shape:", output_tensor.shape)  
    print("Class prediction shape:", class_prediction.shape)  