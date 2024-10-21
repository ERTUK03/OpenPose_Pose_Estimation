from model.feature_extractor import FeatureExtractor
from model.stage import Stage
import torch

class OpenPose(torch.nn.Module):
    def __init__(self, input, num_stages, num_parts, num_connections):
        super(OpenPose, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.pafs = self.get_stages(num_stages, input, input+num_connections*2, num_connections*2)
        self.confs = self.get_stages(num_stages, input+num_connections*2, input+num_parts, num_parts)

    def get_stages(self, stage_nums, input_a, input_b, output):
        stages = [Stage(input_a, output)]
        for i in range(stage_nums-1):
            stages.append(Stage(input_b, output))

        return torch.nn.ModuleList(stages)

    def forward(self, x):
        features = self.feature_extractor(x)
        pafs = []
        for i in range(len(self.pafs)):
            if i == 0:
                pafs.append(self.pafs[i](features))
            else:
                pafs.append(self.pafs[i](torch.cat((pafs[-1], features), 1)))
        confs = []
        for i in range(len(self.confs)):
            if i == 0:
                confs.append(self.confs[i](torch.cat((pafs[-1], features), 1)))
            else:
                confs.append(self.confs[i](torch.cat((confs[-1], features), 1)))
        shapes = pafs[0].shape

        for i in range(len(pafs)):
            paf = pafs[i].view(shapes[0], shapes[1] // 2, shapes[2], shapes[3], 2)
            pafs[i] = paf
        return confs, pafs
