# Ambient Intelligence Project computer vision

### Head and hands recognition

To (re)generate grpc code:
```bash
python -m grpc_tools.protoc -I./grpc_files --python_out=./grpc_files --grpc_python_out=./grpc_files ./grpc_files/image.proto   
```