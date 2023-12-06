import torch
import torch.nn as nn
import torchvision.models

from config import Config

class VAPNet(nn.Module):
    def __init__(self, cfg):
        super(VAPNet, self).__init__()
        self.cfg = cfg

        self.backbone = self.build_backbone(pretrained=True)

        self.spp_pool_size = [5, 2, 1]
        self.adjustment_count = self.cfg.adjustment_count
        
        self.last_layer = nn.Sequential(
            nn.Linear(38400, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.suggestion_output_layer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self.adjustment_output_layer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, self.adjustment_count),
            nn.Softmax(dim=1)
        )

        self.magnitude_output_layer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, self.adjustment_count),
        )
        
    def forward(self, image):
        feature_map = self.backbone(image)
        spp = self.spatial_pyramid_pool(feature_map, feature_map.shape[0], self.spp_pool_size)
        feature_vector = self.last_layer(spp)

        suggestion_predictor = self.suggestion_output_layer(feature_vector)
        adjustment_predictor = self.adjustment_output_layer(feature_vector)
        magnitude_predictor = self.magnitude_output_layer(feature_vector)

        return suggestion_predictor, adjustment_predictor, magnitude_predictor
    
    def build_backbone(self, pretrained):
        model = torchvision.models.mobilenet_v2(pretrained)
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules)
        return backbone
    
    # parameter: tensor, batch_size, tensor width and height, spp pool size
    def spatial_pyramid_pool(self, previous_conv, num_sample, out_pool_size):
        for i in range(len(out_pool_size)):
            maxpool = nn.AdaptiveMaxPool2d((out_pool_size[i], out_pool_size[i]))
            x = maxpool(previous_conv)
            if i == 0:
                spp = x.view([num_sample, -1])
            else:
                spp = torch.cat((spp, x.view([num_sample, -1])), 1)
        return spp
    
if __name__ == '__main__':
    cfg = Config()
    model = VAPNet(cfg)
    model.eval()
    x = torch.randn((1, 3, 224, 224))
    output = model(x)
    print(output)
