sudo docker build -t model-worker .
sudo docker run -d --env-file ./.env --gpus all -it model-worker:latest