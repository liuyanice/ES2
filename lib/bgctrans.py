import torch
import torch.nn as nn
import torch.nn.functional as F
from turtle import forward
from lib.position_embedding import positionEmbeddingLearned
from lib.modules import fusionmodule, boundary_guided_attention
from lib.vision_transformers import cross_scale_transformer
from lib.sampling_points import sampling_points, point_sample
from lib.pvtv2 import pvt_v2_b2  


def load_pvtv2(num_classes, ne_num, md_num, fusion, size): 
    backbone = pvt_v2_b2(img_size = size)
    path = r'root/backbone/pvt_v2_b2.pth'
    save_model = torch.load(path)
    model_dict = backbone.state_dict()
    state_dict = {
        k: v
        for k, v in save_model.items() if k in model_dict.keys()
    }
    model_dict.update(state_dict)
    backbone.load_state_dict(model_dict)
    classifier = linear_classifier(num_classes)
    pixelEncoderModel_pleb = pixelEncoderModel()
    pointHead_bcm = pointHead()
    model = bgcSegmentationModel(backbone, classifier, ne_num, md_num,
                                     fusion, pixelEncoderModel_pleb, pointHead_bcm)
    return model


class linear_classifier(nn.Module):
    def __init__(self, num_classes):
        super(linear_classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(192, 64, 1, padding=1, bias=False),  #560
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1))
        self.classifier1 = nn.Sequential(nn.Conv2d(128, num_classes, 1))
        self.classifier2 = nn.Sequential(nn.Conv2d(128, num_classes, 1))
        self.classifier3 = nn.Sequential(nn.Conv2d(128, num_classes, 1))

    def forward(self, c):
        low_level_feature = c[0]
        output_feature = c[1]
        output_feature = F.interpolate(output_feature,
                                       size=low_level_feature.shape[2:],
                                       mode='bilinear',
                                       align_corners=False)
        if self.training:
            return [
                self.classifier(
                    torch.cat([low_level_feature, output_feature], dim=1)),
                self.classifier1(c[1]),
                self.classifier2(c[2]),
                self.classifier3(c[3])
            ]
        else:
            return self.classifier(
                torch.cat([low_level_feature, output_feature], dim=1))
                
                
class pointHead(nn.Module):
    def __init__(self, in_c=192, num_classes=128, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Conv1d(in_c, num_classes, 1)
        self.k = k
        self.beta = beta

    def forward(self, x, res2, out):
        points = sampling_points(out, x.shape[-1] // 16, self.k, self.beta)
        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(res2, points, align_corners=False)
        feature_representation = torch.cat([coarse, fine], dim=1)
        rend = self.mlp(feature_representation)
        return rend, points


class pixelEncoderModel(nn.Module):
    def __init__(self, features = 64, out_features = 128, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes)+1 ), out_features, kernel_size=1)
        self.relu = nn.ReLU()
 
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)
 
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages]  + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)
              
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
        

class bgcSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, ne_num, md_num, fusion, pixelEncoderModel, pointHead):
        super(bgcSegmentationModel, self).__init__()
        self.backbone = backbone
        self.pixelEncoderModel = pixelEncoderModel
        self.pointHead=pointHead
        self.sampling_points = sampling_points
        self.point_sample = point_sample
        self.classifier = classifier
        self.cross_trans = cross_learner(hidden_features=128,
                                      ne_num=ne_num,
                                      md_num=md_num,
                                      fusion=fusion)                        
        self.classifier4 = nn.Sequential(nn.Conv2d(128, 1, 1))
        self.classifier5 = nn.Sequential(nn.Conv2d(256, 128, 1))
        self.classifier6 = nn.Sequential(nn.Conv2d(192, 128, 1))
        self.boundary_guided_attention = boundary_guided_attention()
        self.ca = ChannelAttention(128)
        self.sa = SpatialAttention()
    def forward(self, x):
        input_shape = x.shape[-2:]
        x_hat = self.backbone(x)  
        I_star = self.pixelDecoderModel(x_hat[0]) 
        I1_star = self.ca(I_star) * I_star + I_star # channel attention
        I2_star = self.sa(I1_star) * I1_star + I1_star # spatial attention
        I3_star = I1_star + I2_star
        y = torch.cat((I3_star, I_star),1)
        I4=self.classifier5(y)
        c, c1_hat, o1 = self.cross_trans(x_hat)

        c1_hat=F.interpolate(c1_hat,
                                    size=(c[0]).shape[-2:],
                                    mode='bilinear',
                                    align_corners=False)                          
        y=torch.cat((I4, c1_hat),1)
        I5=self.classifier5(y)
        crossoutput=self.boundary_guided_attention(I5, o1) 
        m= crossoutput 
        outputs = self.classifier(c)
        rend, points=self.pointHead(x,c[0],m)
        m = self.classifier4(m)  
        m = F.interpolate(m,
                                    size=input_shape,
                                    mode='bilinear',
                                    align_corners=False)
        if self.training:
            outputs = [
                F.interpolate(o,
                              size=input_shape,
                              mode='bilinear',
                              align_corners=False) for o in outputs
            ]
        else:
            outputs = F.interpolate(outputs,
                                    size=input_shape,
                                    mode='bilinear',
                                    align_corners=False)
                                     
        return outputs, m, rend, points


class cross_learner(nn.Module): 
    def __init__(self, hidden_features=128, ne_num=2, md_num=2, fusion=True):
        super().__init__()
        self.ne_num = ne_num
        self.md_num = md_num
        self.conv_1 = nn.Conv2d(in_channels=128,
                                               out_channels=hidden_features,
                                               kernel_size=(1, 1),
                                               stride=(1, 1),
                                               padding=(0, 0),
                                               bias=True)
        self.conv_2 = nn.Conv2d(in_channels=320,
                                               out_channels=hidden_features,
                                               kernel_size=(1, 1),
                                               stride=(1, 1),
                                               padding=(0, 0),
                                               bias=True)
        self.conv_3 = nn.Conv2d(in_channels=512,
                                               out_channels=hidden_features,
                                               kernel_size=(1, 1),
                                               stride=(1, 1),
                                               padding=(0, 0),
                                               bias=True)
        normalize_before = True

        if ne_num + md_num > 0: 
            self.trans1 = cross_scale_transformer(
                point_pred_layers=1,
                num_encoder_layers=ne_num,
                num_decoder_layers=md_num,
                d_model=hidden_features,
                nhead=8,
                normalize_before=normalize_before)
            self.trans2 = cross_scale_transformer(
                point_pred_layers=1,
                num_encoder_layers=ne_num,
                num_decoder_layers=md_num,
                d_model=hidden_features,
                nhead=8,
                normalize_before=normalize_before)
            self.trans3 = cross_scale_transformer(
                point_pred_layers=1,
                num_encoder_layers=ne_num,
                num_decoder_layers=md_num,
                d_model=hidden_features,
                nhead=8,
                normalize_before=normalize_before)
        #self.fusion = fusion
        self.cross_fusion_1 = fusionmodule(hidden_features, 8)
        self.cross_fusion_2 = fusionmodule(hidden_features, 8)
    def forward(self, x):
        features_1 = x[1] #([8, 128, 32, 32])
        features_2 = x[2] #([8, 320, 16, 16])
        features_3 = x[3] # ([8, 512, 8, 8])
        x_hat_1 = self.conv_1(features_1)
        x_hat_2 = self.conv_2(features_2)
        x_hat_3 = self.conv_3(features_3)
        if self.ne_num + self.md_num > 0:  
            o_1, c_1 = self.trans1(x_hat_1)
            o_2, c_2 = self.trans2(x_hat_2)
            o_3, c_3 = self.trans3(x_hat_3)
            if self.md_num > 0:
                o_1 = o_1.permute(2, 0, 1)
                o_2 = o_2.permute(2, 0, 1)
                o_3 = o_3.permute(2, 0, 1)
        else:
            c_1 = features_1
            c_2 = features_2
            c_3 = features_3

        c_hat_2 = self.cross_fusion_2(c_2, c_3, o_2, o_3)
        c_hat_1 = self.cross_fusion_1(c_1, c_hat_2, o_1, o_2)   
        auxiliary_maps = [x[0], c_hat_1, c_hat_2, c_3]
        return auxiliary_maps, c_hat_1, o_1 


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = load_pvtv2().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    prediction1 = model(input_tensor)
