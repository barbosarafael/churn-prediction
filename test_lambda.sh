#!/bin/bash

# Configurações
MODEL_ID="cdeb67983fb547a398617fe30b3c58ce"
PORT=9000
CONTAINER_NAME="churn-prediction-test"

# Função para verificar porta
check_port() {
    if lsof -i :${PORT} > /dev/null; then
        echo "ERRO: A porta ${PORT} já está em uso. Processos:"
        lsof -i :${PORT}
        echo "Soluções:"
        echo "1) Execute 'kill -9 <PID>' para terminar o processo"
        echo "2) Mude a variável PORT no script"
        exit 1
    fi
}

# 1. Verifica se o usuário tem permissão
if ! groups | grep -q '\bdocker\b'; then
    echo "ERRO: Seu usuário não está no grupo docker."
    echo "Execute estes comandos e depois reinicie seu terminal:"
    echo "  sudo usermod -aG docker $USER"
    echo "  newgrp docker"
    exit 1
fi

# Verifica se a porta está disponível
check_port

# 2. Constrói a imagem Docker
docker build -t churn-prediction-lambda --build-arg MODEL_ID=${MODEL_ID} . || {
    echo "ERRO ao construir a imagem Docker";
    exit 1;
}

# 3. Remove container existente (se houver)
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# 4. Inicia o container
docker run -d -p ${PORT}:8080 --name ${CONTAINER_NAME} churn-prediction-lambda || {
    echo "ERRO ao iniciar o container";
    echo "Se o erro for relacionado à porta, tente:"
    echo "1) Fechar aplicações usando a porta 9000"
    echo "2) Executar 'sudo lsof -i :9000' para identificar processos"
    echo "3) Executar 'sudo kill -9 <PID>' para liberar a porta"
    exit 1;
}

# 5. Aguarda o container estar pronto
echo "Aguardando o container iniciar..."
sleep 5

# 6. Testa a API com curl
echo "Testando a API..."
curl -XPOST "http://localhost:${PORT}/2015-03-31/functions/function/invocations" -d '{
    "gender":"Female",
    "Partner":"Yes",
    "Dependents":"No",
    "tenure":24,
    "InternetService":"No",
    "OnlineSecurity":"No internet service",
    "OnlineBackup":"No internet service",
    "DeviceProtection":"No internet service",
    "TechSupport":"No internet service",
    "StreamingTV":"No internet service",
    "StreamingMovies":"No internet service",
    "Contract":"Month-to-month",
    "PaymentMethod":"Mailed check",
    "MonthlyCharges":65.75,
    "TotalCharges":137.6,
    "QtdProducts":0
}'

# 7. Limpeza
echo "Parando o container..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true