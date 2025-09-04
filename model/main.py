import os
import torch
from torch.utils.data import DataLoader
from edgeattnet_model import   # Replace ResNet with your actual model class, e.g., UNet, CNN, etc.
from dataset import CustomDataset  # Replace CustomDataset with your dataset class
from evaluation import evaluate_model, visualize_prediction_by_filename, generate_all_boxplots_in_row  # Replace with your actual functions

def main():
    # Check if we can use the GPU, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up parameters
    batch_size = 8
    test_data_path = './data/test/'  # Path to your test data
    model_path = './models/best_model.pth'  # Path to the model we want to load

    # Initialize the model
    model = ResNet()  # Initialize the model (replace ResNet with your actual model)
    model.to(device)

    # Load the trained weights if they exist
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Couldn’t find the model at {model_path}. Exiting.")
        return

    # Prepare the test data
    test_dataset = CustomDataset(root=test_data_path, mode="test")  # Load your dataset (replace CustomDataset with your actual dataset class)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Run the evaluation
    print("Evaluating the model...")
    results = evaluate_model(model, test_loader, device)
    print(f"Evaluation results: {results}")

    # Visualize a specific prediction by filename (just for example)
    filename = "040401-20220714185352Th"  # Change to any filename from your dataset
    print(f"Visualizing prediction for {filename}...")
    visualize_prediction_by_filename(model, test_dataset, device, filename)

    # If you want, you can generate some boxplots for the results
    print("Generating boxplots...")
    generate_all_boxplots_in_row(
        model=model,
        dataloader=test_loader,
        device=device,
        max_images=12,
        save_name="boxplots_output.pdf"  # You can adjust the file name
    )

if __name__ == "__main__":
    main()
