import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Define convolutional layers for query, key, and value
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Define convolutional layer for attention scores
        self.attention_conv = nn.Conv2d(out_channels, 1, kernel_size=1)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, features):
        # Query, key, and value transformations
        query = self.query_conv(features)
        key = self.key_conv(features)
        value = self.value_conv(features)
        
        # Compute attention scores
        attention_scores = self.attention_conv(torch.tanh(query + key))
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to the value
        attended_features = value * attention_weights
        
        return attended_features

class RangeNetWithFPN(nn.Module):#
    """
    Semantic Segmentation Network with Feature Pyramid Network (FPN) using a ResNet backbone.

    Args:
        backbone (str): Type of ResNet model to use. Supported types: 'resnet18', 'resnet34', 'resnet50', 'resnet101'.
        meta_channel_dim (int): Number meta channels used in the FPN
        interpolation_mode (str): Interpolation mode used to resize the meta channels (default='nearest'). Supported types: 'nearest', 'bilinear', 'bicubic'.
        num_classes (int): Number of semantic classes
        no_bn (bool): Option to disable batchnorm (not recommended)
        attention (bool): Option to use a attention mechanism in the FPN (see: https://arxiv.org/pdf/1706.03762.pdf)
        
    """
    def __init__(self, backbone='resnet18', meta_channel_dim=3, interpolation_mode = 'nearest', num_classes = 2, attention=True, pretrained=False):
        super(RangeNetWithFPN, self).__init__()

        self.backbone_name = backbone
        self.interpolation_mode = interpolation_mode
        self.num_classes = num_classes
        self.attention = attention
        # Load pre-trained ResNet model
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            base_channel = 512  # Number of channels in the last layer of ResNet18
            base_channels = [base_channel, base_channel // 2, base_channel // 4, base_channel // 8, base_channel // 16]
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            base_channel = 512  # Number of channels in the last layer of ResNet34
            base_channels = [base_channel, base_channel // 2, base_channel // 4, base_channel // 8, base_channel // 16]
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            base_channel = 2048  # Number of channels in the last layer of ResNet50
            base_channels = [base_channel, base_channel // 2, base_channel // 4, base_channel // 8, base_channel // 16]
        elif backbone == 'regnet_y_400mf':
            self.backbone = models.regnet_y_400mf(pretrained=pretrained)
            base_channels = [440, 208, 104, 48, 32]
        elif backbone == 'regnet_y_800mf':
            self.backbone = models.regnet_y_800mf(pretrained=pretrained)
            base_channels = [784, 320, 144, 64, 32]
        elif backbone == 'regnet_y_1_6gf':
            self.backbone = models.regnet_y_1_6gf(pretrained=pretrained)
            base_channels = [888, 336, 120, 48, 32]
        elif backbone == 'regnet_y_3_2gf':
            self.backbone = models.regnet_y_3_2gf(pretrained=pretrained)
            base_channels = [1512, 576, 216, 72, 32]
        elif backbone == 'shufflenet_v2_x0_5':
            self.backbone = models.shufflenet_v2_x0_5(pretrained=pretrained)
            base_channels = [1024, 192, 96, 48, 24]
        elif backbone == 'shufflenet_v2_x1_0':
            self.backbone = models.shufflenet_v2_x1_0(pretrained=pretrained)
            base_channels = [1024, 464, 232, 116, 24]
        elif backbone == 'shufflenet_v2_x1_5':
            self.backbone = models.shufflenet_v2_x1_5(pretrained=pretrained)
            base_channels = [1024, 704, 352, 176, 24]
        elif backbone == 'shufflenet_v2_x2_0':
            self.backbone = models.shufflenet_v2_x2_0(pretrained=pretrained)
            base_channels = [2048, 976, 488, 244, 112]
        else:
            raise ValueError("Invalid ResNet type. Supported types: 'resnet18', 'resnet34', 'resnet50', 'regnet_y_400mf','regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0.")
        
        # Modify the first convolution layer to take 1+meta_channel_dim channels
        self.meta_channel_dim = meta_channel_dim

        is_shuffle = False
        # extract features from resnet family
        if backbone in ['resnet18', 'resnet34', 'resnet50']:
            self.backbone.conv1 = nn.Conv2d(3 + meta_channel_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)

            # Extract feature maps from different layers of ResNet
            self.stem = nn.Sequential(self.backbone.conv1, self.backbone.relu, self.backbone.maxpool)
            self.layer1 = self.backbone.layer1
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4
        
        # extract features from regnet family
        elif backbone in ['regnet_y_400mf','regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf']:
            self.backbone.stem[0] = nn.Conv2d(3 + meta_channel_dim, self.backbone.stem[0].out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            # Extract feature maps from different layers of RegNet
            self.stem = self.backbone.stem
            self.layer1 = self.backbone.trunk_output[0]
            self.layer2 = self.backbone.trunk_output[1]
            self.layer3 = self.backbone.trunk_output[2]
            self.layer4 = self.backbone.trunk_output[3]

        elif backbone in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"]:
            self.backbone.conv1[0] = nn.Conv2d(3 + meta_channel_dim, self.backbone.conv1[0].out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            #help_conv = nn.Conv2d(2 + meta_channel_dim, self.backbone.conv1[0].out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            # Extract feature maps using indices or named stages from ShuffleNet
            self.stem = self.backbone.conv1
            self.layer1 = self.backbone.stage2
            self.layer2 = self.backbone.stage3
            self.layer3 = self.backbone.stage4
            self.layer4 = self.backbone.conv5
            is_shuffle = True

        # Attention blocks
        self.attention4 = AttentionModule(base_channels[1], base_channels[1])
        self.attention3 = AttentionModule(base_channels[2], base_channels[2])
        self.attention2 = AttentionModule(base_channels[3], base_channels[3])
        self.attention1 = AttentionModule(base_channels[4], base_channels[4])
        

        # FPN blocks
        self.fpn_block4 = self._make_fpn_block(base_channels[0], base_channels[1])
        self.fpn_block3 = self._make_fpn_block(base_channels[1], base_channels[2])
        self.fpn_block2 = self._make_fpn_block(base_channels[2], base_channels[3])
        self.fpn_block1 = self._make_fpn_block(base_channels[3], base_channels[4])

        # upsamle layers
        if is_shuffle:
            self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels[1], out_channels=base_channels[1], kernel_size=4, stride=4, padding=0)
        else:
            self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels[1], out_channels=base_channels[1], kernel_size=8, stride=8, padding=0)
        self.upsample_layer_x3 = nn.ConvTranspose2d(in_channels=base_channels[2], out_channels=base_channels[2], kernel_size=4, stride=4, padding=0)
        self.upsample_layer_x2 = nn.ConvTranspose2d(in_channels=base_channels[3], out_channels=base_channels[3], kernel_size=2, stride=2, padding=0)
        

        # self.decoder_semantic = nn.Sequential(
        #     nn.Conv2d(base_channels[1] + base_channels[2] + base_channels[3] + base_channels[4], base_channels[4], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(base_channels[4]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(base_channels[4], base_channels[4], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(base_channels[4]),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(base_channels[4], self.num_classes, kernel_size=4, stride=2, padding=1),
        # )

        self.decoder_range = nn.Sequential(
            nn.Conv2d(base_channels[1] + base_channels[2] + base_channels[3] + base_channels[4], base_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels[4], base_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels[4]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels[4], 1, kernel_size=4, stride=2, padding=1),
        )

    def _make_fpn_block(self, in_channels, out_channels):
        """
        Create an FPN block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Module: FPN block.
        """

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
            
    def forward(self, x, meta_channel):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing depth prediction.
        """
        
        
        # Inject meta channel before ResNet layers
        if self.meta_channel_dim > 0:
            # Resize Meta Channels
            # Downsample the meta channel
            meta_channel1 = F.interpolate(meta_channel, scale_factor=1/2, mode=self.interpolation_mode)
            meta_channel2 = F.interpolate(meta_channel, scale_factor=1/4, mode=self.interpolation_mode)
            meta_channel3 = F.interpolate(meta_channel, scale_factor=1/8, mode=self.interpolation_mode)
            x = torch.cat([x, meta_channel], dim=1)
            xs = self.stem(x)
            x1 = self.layer1(xs)
            x = torch.cat([x1[:,0:-self.meta_channel_dim,...], meta_channel1], dim=1)
            x2 = self.layer2(x)
            x = torch.cat([x2[:,0:-self.meta_channel_dim,...], meta_channel2], dim=1)
            x3 = self.layer3(x)
            x = torch.cat([x3[:,0:-self.meta_channel_dim,...], meta_channel3], dim=1)
            x4 = self.layer4(x)

        else:
        
            # Encoder (ResNet)
            xs = self.stem(x)
            x1 = self.layer1(xs)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)

        # FPN
        x4 = self.fpn_block4(x4)
        x3 = self.fpn_block3(x3)
        x2 = self.fpn_block2(x2)
        x1 = self.fpn_block1(x1)
        
        # Attention
        if self.attention:
            x4 = self.attention4(x4)
            x3 = self.attention3(x3)
            x2 = self.attention2(x2)
            x1 = self.attention1(x1)

        

        x4 = self.upsample_layer_x4(x4)
        x3 = self.upsample_layer_x3(x3)
        x2 = self.upsample_layer_x2(x2)

        # Concatenate feature maps
        x = torch.cat([x1, x2, x3, x4], dim=1)


        # Decoder
        x_range = self.decoder_range(x)
        #x_semantics = self.decoder_semantic(x)
        return x_range#, x_semantics
    