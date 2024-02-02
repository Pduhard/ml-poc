import torch
import torch.nn as nn

class RoomPopulationModel2(nn.Module):
    def __init__(self, num_object_types, max_objects=50):
        super(RoomPopulationModel2, self).__init__()
        
        # Define the architecture for processing room and object bounding boxes
        self.bb_processing = nn.Sequential(
            nn.Linear(408, 512, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(512, 1024, dtype=torch.float64),
            nn.ReLU()
        )
        
        # Define the architecture for processing object types
        self.type_processing = nn.Sequential(
            nn.Linear(num_object_types * max_objects, 512, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(512, 1024, dtype=torch.float64),
            nn.ReLU(),
        )
        
        # Define the final fully connected layers for detection
        self.detection_layers = nn.Sequential(
            nn.Linear(2048, 4096, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(4096, 1024, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(1024, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, (max_objects + 1) * 8, dtype=torch.float64)  # Each object has 4 coordinates and num_object_types for one-hot encoding
        )

    def forward(self, room_bb, object_bbs, object_types):
        # Process room and object bounding boxes
        object_feats = self.bb_processing(torch.cat([room_bb, object_bbs], dim=1))  # Flatten the object_bbs tensor
        # Process object types
        type_feats = self.type_processing(object_types)
        # Concatenate features
        combined_feats = torch.cat([object_feats, type_feats], dim=1)
        # Final detection layers
        detection_out = self.detection_layers(combined_feats)
        
        return detection_out

    def save(self, path):
        torch.save(self.state_dict(), path)