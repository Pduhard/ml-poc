import torch
import torch.nn as nn

class RoomPopulationModel2(nn.Module):
    def __init__(self, num_object_types, max_objects=50):
        super(RoomPopulationModel2, self).__init__()
        
        self.bb_processing = nn.Sequential(
            nn.Linear(408, 512, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(512, 1024, dtype=torch.float32),
            nn.ReLU()
        )
        
        self.type_processing = nn.Sequential(
            nn.Linear(num_object_types * max_objects, 512, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(512, 1024, dtype=torch.float32),
            nn.ReLU(),
        )
        
        self.detection_layers = nn.Sequential(
            nn.Linear(2048, 1024, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1024, 256, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(256, (max_objects + 1) * 8 * 7, dtype=torch.float32)
        )

        self.max_objects = max_objects

    def forward(self, x):
        # print(x.dtype)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        object_feats, object_types = x[..., :8 + 8 * self.max_objects], x[..., 8 + 8 * self.max_objects:]

        # print(object_feats.dtype)
        # print(object_types.dtype)
        object_feats = self.bb_processing(object_feats)
        type_feats = self.type_processing(object_types)

        combined_feats = torch.cat([object_feats, type_feats], dim=1)
        # print(combined_feats.dtype)

        detection_out = self.detection_layers(combined_feats)
        # print(detection_out.dtype)
        return detection_out

    def save(self, path):
        torch.save(self.state_dict(), path)