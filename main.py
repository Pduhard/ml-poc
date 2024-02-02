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
        os.path.join("/home/paco/dev/Kazaplan/poc_ai_population/plan_csv2"),
        os.path.join("/home/paco/dev/Kazaplan/poc_ai_population/pfLabels.csv"),
    )

    model = RoomPopulationModel2(
        num_object_types=len(dataset.all_labels),
    )
 
    train_data, test_data = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        total_loss = 0
        for batch, (train_features, train_targets) in enumerate(train_dataloader):
            # Compute prediction error
            pred = model(train_features)
            loss = torch.nn.functional.mse_loss(pred, train_targets)
            total_loss += loss.item()
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1} train: {total_loss / len(train_dataloader)} \n-------------------------------')
        total_loss = 0


        # validation
        model.eval()
        with torch.no_grad():
            for batch, (test_features, test_targets) in enumerate(test_dataloader):
                # Compute prediction error
                pred = model(test_features)
                loss = torch.nn.functional.mse_loss(pred, test_targets)
                total_loss += loss.item()

        print(f'Epoch {epoch + 1} test: {total_loss / len(test_dataloader)} \n-------------------------------')

        # if (total_loss / len(test_dataloader)) / (total_loss / len(train_dataloader)) < 0.8:
        #     model.save(f'model-${str(epoch)}-ratio-${str(total_loss / len(test_dataloader)) / (total_loss / len(train_dataloader))}.pt')
    print('Done!')
    model.save("model.pt")

    dummy_input = torch.randn(dataset[0][0].shape[0], dtype=torch.float32)
    print(dummy_input.shape)

    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "Network-translated-room.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=17,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={
            'input' : {0 : 'batch_size'},    # variable length axes 
            'output' : {0 : 'batch_size'}
        }
    )
    