import torch
import torch.nn as nn

# 网络结构参照的原作者，虽然有点奇怪
cfg = {
    # kernel, padding, channel, stride
    'TextBoxes': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'B', 'M', 512, 512, 512, 'M', \
            (3, 6, 1024, 1), (1, 0, 1024, 1), 'B', \
            (1, 0, 256, 1), (3, 1, 512, 2), 'B', \
            (1, 0, 128, 1), (3, 1, 256, 2), 'B', \
            (1, 0, 128, 1), (3, 1, 256, 2), 'B', \
            'G', 'BE']
}

class TextBoxes(nn.Module):
    def __init__(self, cfg, init_weights=True):
        super(TextBoxes, self).__init__()
        self.cfg = cfg
        self.bone, self.boxes = self.make_layers(cfg)
        self.bone = nn.ModuleList(self.bone)
        self.boxes = nn.ModuleList(self.boxes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out_boxes = None
        textboxes = cfg['TextBoxes']
        bone_index = 0
        boxes_index = 0
        for v in textboxes:
            if isinstance(v, str) and (v == 'B' or v == 'BE'):
                box = self.boxes[boxes_index](x)
                box = box.permute(2, 3, 0, 1)
                _, _, batch, channel = box.size()
                box = box.reshape(-1, batch, channel)
                if out_boxes is None:
                    out_boxes = box
                else:
                    out_boxes = torch.cat((out_boxes, box), 0)
                boxes_index += 1
            else:
                x = self.bone[bone_index](x)
                bone_index += 1
        # change to batch, local, confidence and boxes
        out_boxes = out_boxes.permute(1, 0, 2)
        return out_boxes

    def make_layers(self, cfg):
        textboxes = cfg['TextBoxes']
        bone = []
        boxes = []
        in_channels = 3
        for v in textboxes:
            if isinstance(v, str)  and (v == 'B' or v == 'BE'):
                boxes += self.get_layer(v, in_channels)
            else:
                layer, in_channels = self.get_layer(v, in_channels)
                bone += layer
        return bone, boxes

    def get_layer(self, v, in_channels):
        if isinstance(v, tuple):
            return [nn.Conv2d(in_channels, v[2], kernel_size=v[0], padding=v[1], stride=v[3])], v[2]
        if v == 'M':
            return [nn.MaxPool2d(kernel_size=2, stride=2)], in_channels
        elif v == 'B':
            return [nn.Conv2d(in_channels, 72, kernel_size=(1, 5), stride=1, padding=(0, 2))]
        elif v == 'BE':
            return [nn.Conv2d(in_channels, 72, kernel_size=1)]
        elif v == 'G':
            return [nn.AdaptiveAvgPool2d((1, 1))], in_channels
        else:
            return [nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)], v

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
if __name__ == "__main__":
    tbox = TextBoxes(cfg).to("cuda")
    x = torch.randn(10, 3, 300, 300).to("cuda")
    out = tbox(x)
    print(len(out))
    print(out.shape)