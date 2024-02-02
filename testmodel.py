import torch
from model import RoomPopulationModel  # Make sure this import matches the definition of your model

# Define the path to your model file
model_path = 'model.pt'

# Assuming the model's architecture is defined in `RoomPopulationModel`
# Update the input and output size if they differ
model = RoomPopulationModel(input_size=3408, output_size=1608)

# Load the model parameters from the .pt file
model.load_state_dict(torch.load(model_path))

# Ensure the model is in evaluation mode
model.eval()

# Prepare your input data in the same format as your training data
# Here you need to convert your input data to a PyTorch Tensor
# This is an example based on your provided data format
# You might need to adjust it according to the actual input your model expects

# Example input tensor initialization (replace this with your actual input)
# For the sake of demonstration, we're using random data
# The shape of the tensor should match the input shape expected by your model
input_tensor = torch.randn(1, 3408, dtype=torch.float64)  # Adjust the shape accordingly

# Use the model to make predictions
with torch.no_grad():
    prediction = model(input_tensor)

# Process the prediction as needed
print(prediction)
