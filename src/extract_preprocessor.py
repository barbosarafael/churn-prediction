import mlflow
import joblib
from pathlib import Path

# Caminho para o modelo
id_best_model = 'cdeb67983fb547a398617fe30b3c58ce'
model_path = f"models/best_model_{id_best_model}/model.pkl"
save_dir = f'calibrated_models/calibrated_model_{id_best_model}/'

try:
    # Tenta carregar como pipeline do sklearn primeiro
    model = joblib.load(model_path)
    
    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        preprocessor = model.named_steps['preprocessor']
        output_path = Path(f"{save_dir}preprocessor_{id_best_model}.pkl")
        joblib.dump(preprocessor, output_path)
        print(f"✅ Pré-processador extraído e salvo em: {output_path}")
        print(f"Tipo do pré-processador: {type(preprocessor)}")
        print(f"Steps: {getattr(preprocessor, 'named_steps', 'Não é um Pipeline')}")
    else:
        print("⚠️ O modelo não contém um pré-processador como step nomeado")
        print("Estrutura do modelo encontrada:")
        print(dir(model))
        
except Exception as e:
    print(f"❌ Erro ao extrair pré-processador: {str(e)}")
    print("Tentando carregar como modelo MLflow...")
    
    try:
        model = mlflow.pyfunc.load_model(model_path)
        if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'named_steps'):
            preprocessor = model._model_impl.named_steps.get('preprocessor')
            if preprocessor:
                output_path = Path("preprocessor.pkl")
                joblib.dump(preprocessor, output_path)
                print(f"✅ Pré-processador extraído via MLflow e salvo em: {output_path}")
            else:
                print("ℹ️ O modelo MLflow não contém um pré-processador")
        else:
            print("ℹ️ Estrutura do modelo MLflow não contém steps nomeados")
    except Exception as e2:
        print(f"❌ Falha ao carregar como modelo MLflow: {str(e2)}")