import tensorrt as trt

#Logger对象
TRT_LOGGER=trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def onnx2trt(onnx_path,engine_path):
    with trt.Builder(TRT_LOGGER) as builder,\
    builder.create_network(EXPLICIT_BATCH) as network,\
    trt.OnnxParser(network,TRT_LOGGER)as parser:
        #builder.max_workspace_size=1<<28 #256MB #tensorRT 8.0版本以上无该函数
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28
        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)
        with open(onnx_path,'rb') as model:
            parser.parse(model.read())
        engine = builder.build_engine(network, config)

        with open(engine_path,'wb')as f:
            f.write(engine.serialize())

if __name__=="__main__":
    onnx_path="./model.onnx"
    engine_path="./model.engine"
    onnx2trt(onnx_path,engine_path)