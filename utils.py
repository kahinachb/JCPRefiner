import yaml
import pandas as pd

def read_subject_yaml(file_path):
    """
    Lit un fichier YAML et retourne directement id, height, weight et gender.
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    subject_id = data.get('id')
    height = data.get('height')
    weight = data.get('weight')
    gender = data.get('gender')
    
    return subject_id, height, weight, gender
def save_to_csv(data, output_path, header=None):
    """Save 3D keypoints to a CSV file with optional header."""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, header=header if header is not None else False)
    print(f"Saved {len(data)} frames to {output_path}")

