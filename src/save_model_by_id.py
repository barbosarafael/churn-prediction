# Import libraries 

import yaml
from utils_modelling import save_model_by_id

# Carregar configurações
with open('parameters.yml', 'r') as f:
    config = yaml.safe_load(f)

MODEL_PATH = config['modelling']['model_path']

# Função principal
def main():
    
    try:
        
        save_model_by_id(run_id = '...', save_dir = MODEL_PATH)
        
    except:
        
        print("Modelo não salvo")

if __name__ == "__main__":
    main()