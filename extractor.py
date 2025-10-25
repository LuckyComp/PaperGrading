from .feature_extraction import FeatureExtractor
from threading import Thread
import sys
import pathlib
import csv
from sentence_transformers import SentenceTransformer, util
import numpy as np

def pipeline(feature_ext:FeatureExtractor) -> list[float|np.typing.NDArray[np.float64]]:
    features = []
    try:
        coherence = feature_ext.get_coherence_score()
        abs_quality = feature_ext.get_abstract_quality()
        features.append(coherence)
        features.append(abs_quality)
    except Exception as e:
        print(f"ERR SCALAR FEATURES: {e}")
        return None # Return None if scalar extraction fails

    try:
        encodings_result = feature_ext.get_encodings()
        features += encodings_result
        return features
    except ValueError as ve: # Catch the specific error
        print(f"ERR FEATURES (ValueError - Ambiguity likely inside get_encodings): {ve}") # Specific message
        return None
    except Exception as e:
        print(f"ERR FEATURES (Other): {e}")
        return None

def save_to_output(path:str, features:list):
    with open(path, "a", newline="") as data_doc:
        writer = csv.writer(data_doc)
        writer.writerow(features)
    print("Added new data tuple successfully.")

def process_files(path:str, output_path:str|None):
    sentence_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    pdf_folder = pathlib.Path(path)

    if not pdf_folder.is_dir():
        print(f"Error: The path '{path}' is not a valid directory.")
        return

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{path}'.")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting processing...\n")

    for i, pdf_path in enumerate(pdf_files):
        try:
            feat_extract = FeatureExtractor(document_name=pdf_path, sen_model=sentence_model)
            features = pipeline(feat_extract)
            if output_path:
                save_to_output(output_path, features)
            else:
                return features
            print(f"Completed file number {i+1}.")
        except Exception as e:
            print(f"Could not open or process {pdf_path.name}. Error: {e}")
        print("-" * 20)

def process_files(pdf_path:str, sentence_model:SentenceTransformer|None=None) -> list[float|np.typing.NDArray[np.float64]]:
    if not sentence_model:
        sentence_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    try:
        feat_extract = FeatureExtractor(document_name=pdf_path, sen_model=sentence_model)
        features = pipeline(feat_extract)
        return features
    except FileNotFoundError as e:
        print(f"Could not open or process {pdf_path.name}. Error: {e}")
    print("-" * 20)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERR: Incorrect command-line arguments.")
        print(sys.argv)
        exit(-1)
    process_files(sys.argv[1], sys.argv[2])
    