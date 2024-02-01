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

    #setup optimizer
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

    for epoch in range(1):

        # print(f'Epoch {epoch + 1}\n-------------------------------')
        # train_data, test_data = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])
        train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        for batch, (train_features, train_targets) in enumerate(train_dataloader):
            train_features, train_targets = train_features.to(device), train_targets.to(device)
            # print(f'batch {batch} : {train_features.shape} -> {train_targets.shape}')

            # Compute prediction error
        #     pred = model(train_features)
        #     loss = torch.nn.functional.mse_loss(pred, train_targets)
        #     print(f'loss: {loss.item()}')

        #     # # Backpropagation
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        # # validation
        # test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)
        # for batch, (test_features, test_targets) in enumerate(test_dataloader):
        #     test_features, test_targets = test_features.to(device), test_targets.to(device)
        #     print(f'batch {batch} : {test_features.shape} -> {test_targets.shape}')

        #     # Compute prediction error
        #     pred = model(test_features)
        #     loss = torch.nn.functional.mse_loss(pred, test_targets)
        #     print(f'loss: {loss.item()}')

    print('Done!')
    model.save("model.pt")