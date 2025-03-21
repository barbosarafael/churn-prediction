# Import libraries 

import yaml
from utils_modelling import save_model_by_id

# Load configurations
with open('parameters.yml', 'r') as f:
    config = yaml.safe_load(f)

MODEL_PATH = config['modelling']['model_path']

# Main function
def main():
    
    try:
        
        save_model_by_id(run_id = '...', save_dir = MODEL_PATH)
        
    except:
        
        print("Model not saved")

if __name__ == "__main__":
    main()
