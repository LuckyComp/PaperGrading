from .trainer import HybridAutoencoder, get_reconstruction_errors
from .extractor import process_files
import sys
from sentence_transformers import SentenceTransformer
from torch import nn
import os
import torch
import pathlib
import numpy as np

def model_predict(feature_vector:list[float|np.typing.NDArray[np.float64]]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridAutoencoder().to(device)
    criterion = nn.MSELoss(reduction='none')
    try:
        project_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(project_dir, "PaperGrading", r"pre-weights\autoencoder_weights.pt")   
        model.load_state_dict(torch.load(data_path, map_location=device))
    except Exception as e:
        print(f"Model not loaded: {e}")
        sys.exit(1)
    model.eval()
    scalars, vectors = (feature_vector[:2], feature_vector[2:])
    
    if None in scalars:
        print(f"Error: Feature extraction failed to get scalar values. Got: {scalars}")
        print("Cannot proceed with prediction. Check the feature extractor's logic on the input file.")
        sys.exit(1)
    
    try:
        # Ensure scalars are floats before converting to tensor
        s_data_tensor = torch.tensor([float(s) for s in scalars], dtype=torch.float32)
    except (ValueError, TypeError) as e:
        print(f"Error converting scalars to tensor: {e}. Scalars: {scalars}")
        sys.exit(1)

    try:
        vector_features_np = np.stack(vectors)
        v_data_tensor = torch.tensor(vector_features_np, dtype=torch.float32)
    except (ValueError, TypeError) as e:
        print(f"Error converting vectors to tensor: {e}.")
        sys.exit(1)
    recon_loss = get_reconstruction_errors(model=model, criterion=criterion,s_data=s_data_tensor,v_data=v_data_tensor,device=device)
    return recon_loss

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"ERR: Incorrect Command-line arguments! {len(sys.argv)}")
        sys.exit(1)
    reject_count, accept_count = (0,0)
    pdf_path_dir = sys.argv[1].replace("\\", "/")
    pdf_folder = pathlib.Path(pdf_path_dir)
    sentence_model = SentenceTransformer("allenai/scibert_scivocab_uncased")

    if not pdf_folder.is_dir():
        print(f"Error: The path '{pdf_path_dir}' is not a valid directory.")
        sys.exit(1)

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{pdf_path_dir}'.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files. Starting processing...\n")

    for i, pdf_path in enumerate(pdf_files):
        threshold = 0.011
        try:
            features = process_files(pdf_path, sentence_model)
            if features is None: # Add check in case process_files failed
                print("ERR: Feature extraction returned None.")
                continue # Skip this fill
            else:
                overall_loss, scalar_errors, vector_errors = model_predict(features)
                print(overall_loss)
                if overall_loss > threshold:
                    print("Rejected")
                    reject_count += 1
                else:
                    print("Accepted")
                    accept_count += 1
                print(f"Completed file name {pdf_path.name}.")
        except Exception as e:
            print(f"Could not open or process {pdf_path.name}. Error: {e}")
        print("-" * 20)
        print(f"Accept: {accept_count}\nReject: {reject_count}")