import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from torch.utils.data import random_split

from model2 import RoomPopulationModel2
from dataset import PopulationDataset


## DATA AUGMENTATION IDEA:
## - reverse the order of the objects in the room


def retrieve_labels_from_input(input_list):
    labels = []
    for index, array in enumerate(input_list):
        labels.append(retrieve_label(array[408:], dataset.get_all_labels()))
    return labels


def retrieve_label(arr, all_labels):
    """
    Retrieves the label for each occurrence of "1.0" in the array,
    looking at every ninth element starting from the current "1.0" element.
    If there are no "1.0" values in the 9-element slice, label it as "unknown".

    :param arr: Input array containing 0.0 and 1.0 values
    :param all_labels: List of labels corresponding to each index
    :return: List of labels identified based on the array
    """
    labels_found = []
    # Iterate over the array
    for i in range(0, len(arr), 9):
        # Check if there is a "1.0" in the 9-element slice
        if 1.0 in arr[i:i + 9]:
            # Find the index of "1.0" in the slice
            index_of_one = arr[i:i + 9].index(1.0)
            # Calculate label index by adding the index of "1.0" to the start index of the slice
            label_index = i + index_of_one
            labels_found.append(all_labels[label_index % len(all_labels)])
        else:
            # If there are no "1.0" values in the slice, label it as "unknown"
            labels_found.append("unknown")

    return labels_found


def format_output_validation_predictions_to_csv(validation_data):
    labels_info = validation_data["all_labels_info"]
    predictions = validation_data["predictions"]

    for index, prediction in enumerate(predictions[0]):
        columns = [
            "label",
            "bounding_box_x1",
            "bounding_box_y1",
            "bounding_box_x2",
            "bounding_box_y2",
            "bounding_box_x3",
            "bounding_box_y3",
            "bounding_box_x4",
            "bounding_box_y4",
        ]
        output_data = []
        # first 8 item of predictions are the bounding box coordinates of "ROOM" label
        room_coordinates = prediction[:8]
        print(room_coordinates)
        output_data.append({
            "label": "ROOM",
            "bounding_box_x1": room_coordinates[0],
            "bounding_box_y1": room_coordinates[1],
            "bounding_box_x2": room_coordinates[2],
            "bounding_box_y2": room_coordinates[3],
            "bounding_box_x3": room_coordinates[4],
            "bounding_box_y3": room_coordinates[5],
            "bounding_box_x4": room_coordinates[6],
            "bounding_box_y4": room_coordinates[7],
        })
        # the rest of the predictions are the bounding box coordinates of the labels 8 by 8
        label_coordinates = prediction[8:]
        # Create a list of dictionaries containing the bounding box coordinates
        for i in range(0, len(label_coordinates), 8):
            try:
                print(i // 8)
                output_data.append({
                    "label": labels_info[0][index][i // 8],
                    "bounding_box_x1": label_coordinates[i],
                    "bounding_box_y1": label_coordinates[i + 1],
                    "bounding_box_x2": label_coordinates[i + 2],
                    "bounding_box_y2": label_coordinates[i + 3],
                    "bounding_box_x3": label_coordinates[i + 4],
                    "bounding_box_y3": label_coordinates[i + 5],
                    "bounding_box_x4": label_coordinates[i + 6],
                    "bounding_box_y4": label_coordinates[i + 7],
                })
            except:
                pass

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(output_data, columns=columns)
        # clean folder "./validation_predictions
        if os.path.exists(f"validation_predictions_"):
            os.remove(f"validation_predictions_")
        os.makedirs(f"validation_predictions", exist_ok=True)
        # Save the DataFrame to a CSV file for each index
        df.to_csv(f"validation_predictions/{index}.csv", index=False)

        print("Validation predictions saved to validation_predictions.csv")


if __name__ == "__main__":
    print(os.path)
    print(torch.__version__)

    dataset = PopulationDataset(
        os.path.join("plan_csv2"),
        os.path.join("pfLabels.csv"),
    )

    model = RoomPopulationModel2(
        num_object_types=len(dataset.all_labels),
    )

    train_data, test_data = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(12):
        total_loss = 0
        for batch, (train_features, train_targets) in enumerate(train_dataloader):
            # Compute prediction error
            pred = model(train_features)
            loss = torch.nn.functional.mse_loss(pred, train_targets)
            total_loss += loss.item()
            print(f'Epoch {epoch + 1} batch {batch} loss: {loss.item()}')
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1} train: {total_loss / len(train_dataloader)} \n-------------------------------')
        total_loss = 0

        # validation
        validation_losses = []
        all_predictions = []
        all_targets = []
        all_labels_info = []

        model.eval()
        with torch.no_grad():
            for batch, (test_features, test_targets) in enumerate(test_dataloader):
                # Compute prediction error
                pred = model(test_features)
                loss = torch.nn.functional.mse_loss(pred, test_targets)
                total_loss += loss.item()

                validation_losses.append(loss.item())
                all_predictions.append(pred.cpu().numpy().tolist())
                all_targets.append(test_targets.cpu().numpy().tolist())
                all_labels_info.append(retrieve_labels_from_input(test_features.tolist()))

        print(f'Epoch {epoch + 1} test: {total_loss / len(test_dataloader)} \n-------------------------------')
        # Prepare the data for saving
        validation_data = {
            "predictions": all_predictions[:409],
            "all_labels_info": all_labels_info,
            "input": test_features.tolist(),
        }

        # Save to JSON
        validation_output_path = 'validation_data.json'  # Adjust path as necessary
        with open(validation_output_path, 'w') as f:
            json.dump(validation_data, f, indent=4)

        format_output_validation_predictions_to_csv(validation_data)

    print('Done!')

    model.save("model.pt")

    dummy_input = torch.randn(dataset[0][0].shape[0], dtype=torch.float32)
    print(dummy_input.shape)

    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "Network-translated-room.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={
                          'input': {0: 'batch_size'},  # variable length axes
                          'output': {0: 'batch_size'}
                      }
                      )
