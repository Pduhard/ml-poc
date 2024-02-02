import os
import torch
from torch.utils.data import DataLoader

from torch.utils.data import random_split

from model2 import RoomPopulationModel2
from dataset import PopulationDataset

## DATA AUGMENTATION IDEA: 
## - reverse the order of the objects in the room

if __name__ == "__main__":
    print(os.path)
    print(torch.__version__)

    dataset =  PopulationDataset(
        os.path.join("/home/paco/dev/Kazaplan/poc_ai_population/plan_csv"),
        os.path.join("/home/paco/dev/Kazaplan/poc_ai_population/pfLabels.csv"),
    )

    model = RoomPopulationModel2(
        num_object_types=len(dataset.all_labels),
    )
 
    train_data, test_data = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

    #setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        # print(f'Epoch {epoch + 1}\n-------------------------------')

        total_loss = 0
        for batch, (train_features, train_targets) in enumerate(train_dataloader):
            # print(f'batch {batch} : {train_features.shape} -> {train_targets.shape}')
            room_bb, object_bbs, object_types = train_features
            # Compute prediction error
            pred = model(room_bb, object_bbs, object_types)
            loss = torch.nn.functional.mse_loss(pred, train_targets)
            total_loss += loss.item()

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1} train: {total_loss / len(train_dataloader)} \n-------------------------------')
        total_loss = 0

        # validation
        for batch, (test_features, test_targets) in enumerate(test_dataloader):
            # Compute prediction error
            room_bb, object_bbs, object_types = test_features
            # Compute prediction error
            pred = model(room_bb, object_bbs, object_types)
            loss = torch.nn.functional.mse_loss(pred, test_targets)
            total_loss += loss.item()

        print(f'Epoch {epoch + 1} test: {total_loss / len(test_dataloader)} \n-------------------------------')

        if (total_loss / len(test_dataloader)) / (total_loss / len(train_dataloader)) < 0.8:
            model.save(f'model-${str(epoch)}-ratio-${str(total_loss / len(test_dataloader)) / (total_loss / len(train_dataloader))}.pt')
    print('Done!')
    model.save("model.pt")

    dummy_input = torch.randn(dataset[0][0].shape[0], dtype=torch.float64)
    torch.onnx.export(model,               # model being run
                  dummy_input,                         # model input (or a tuple for multiple inputs)
                  "population-model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}}
    )