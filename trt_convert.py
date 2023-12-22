import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')
    
    builder.max_workspace_size = 1 << 28  # Adjust this value as per your system's memory
    builder.fp16_mode = False  # Can be set to True for FP16 precision
    engine = builder.build_cuda_engine(network)
    return engine

# Build the engine
engine = build_engine("lidar2camera.onnx")

# Save engine to file
with open("lidar2camera.trt", "wb") as f:
    f.write(engine.serialize())