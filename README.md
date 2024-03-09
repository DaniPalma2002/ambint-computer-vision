# Ambient Intelligence Project computer vision

### Head and hands recognition

To (re)generate grpc code:
```bash
python -m grpc_tools.protoc -I./proto --python_out=./proto --grpc_python_out=./proto ./proto/image.proto   
```