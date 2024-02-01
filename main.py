import os
import torch
from torch.utils.data import DataLoader

from torch.utils.data import random_split

from model import RoomPopulationModel
from dataset import PopulationDataset

if __name__ == "__main__":
    print(os.path)
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    dataset =  PopulationDataset(
        os.path.join("/home/paco/dev/Kazaplan/poc_ai_population/plan_csv"),
        os.path.join("/home/paco/dev/Kazaplan/poc_ai_population/pfLabels.csv"),
    )

    model = RoomPopulationModel(
        input_size=dataset[0][0].shape[0],
        output_size=dataset[0][1].shape[0],
    ).to(device)
    print(dataset[0][0].shape[0])
    print(dataset[0][1].shape[0])

    train_data, test_data = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

    #setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(15):
        # print(f'Epoch {epoch + 1}\n-------------------------------')

        ttloss = 0
        for batch, (train_features, train_targets) in enumerate(train_dataloader):
            train_features, train_targets = train_features.to(device), train_targets.to(device)
            # print(f'batch {batch} : {train_features.shape} -> {train_targets.shape}')

            # Compute prediction error
            pred = model(train_features)
            loss = torch.nn.functional.mse_loss(pred, train_targets)
            print(f'loss: {loss.item()}')

            ttloss += loss.item()

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1} train: {ttloss / len(train_dataloader)} \n-------------------------------')
        ttloss = 0

        # validation
        for batch, (test_features, test_targets) in enumerate(test_dataloader):
            test_features, test_targets = test_features.to(device), test_targets.to(device)

            # Compute prediction error
            pred = model(test_features)
            loss = torch.nn.functional.mse_loss(pred, test_targets)
            loss = torch.nn.functional.mse_loss(pred, test_targets)
            ttloss += loss.item()

        print(f'Epoch {epoch + 1} test: {ttloss / len(test_dataloader)} \n-------------------------------')

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