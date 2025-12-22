import torch
from struct_xlmr import StructXLMRoberta # Your custom model class

# The path to your saved model file
CHECKPOINT_PATH = "struct_roberta_final.pt" 

# Load the checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

# Get the best validation loss that was stored in the file
best_loss = checkpoint.get('stored_loss', 'Not found')

print(f"The model in '{CHECKPOINT_PATH}' was saved with a best validation loss of: {best_loss:.2f}")



# --- Configuration ---
BASE_MODEL_PATH = "Roberta_hi"

print("--- Loading Model Architecture ---")

# Load the model architecture
model = StructXLMRoberta(model_name=BASE_MODEL_PATH)

print("\n" + "="*50)
print("          Model Architecture")
print("="*50)
print(model)

# --- Optional: Print number of parameters ---
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "="*50)
print("          Model Parameters")
print("="*50)
print(f"Total Parameters:     {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print("="*50)