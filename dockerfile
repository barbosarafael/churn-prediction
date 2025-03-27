FROM public.ecr.aws/lambda/python:3.9

# 1. Define o ID do modelo como argumento
ARG MODEL_ID=cdeb67983fb547a398617fe30b3c58ce

# 2. Instala dependências
RUN yum install -y gcc gcc-c++ make openssl-devel libffi-devel python3-devel && \
    pip install --upgrade pip setuptools wheel

# 3. Cria estrutura de diretórios
RUN mkdir -p /opt/ml/model

# 4. Copia TODOS os arquivos do modelo
COPY calibrated_models/calibrated_model_${MODEL_ID}/ /opt/ml/model/

# 5. Cria link simbólico para o arquivo do modelo
RUN ln -s /opt/ml/model/model_${MODEL_ID}.pkl /opt/ml/model/model.pkl && \
    ln -s /opt/ml/model/preprocessor_${MODEL_ID}.pkl /opt/ml/model/preprocessor.pkl

# 6. Instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copia o código
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# 8. Define variável de ambiente
ENV MODEL_ID=${MODEL_ID}

CMD ["src.lambda_handler.lambda_handler"]