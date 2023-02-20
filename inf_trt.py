import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from train import get_data
trt_logger= trt.Logger(trt.Logger.INFO)
trt_runtime=trt.Runtime(trt_logger)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
if __name__=="__main__":
    # load engine 
    with open("./model.engine",'rb') as f:
        engine_data=f.read()
    # 反序列化
    engine=trt_runtime.deserialize_cuda_engine(engine_data)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # 创建执行上下文
    context=engine.create_execution_context()
    train_loader,test_loader = get_data(batch_size=1)
    correct=0
    for x, y in test_loader:
        inputs[0].host=np.array(x)
        [output] =do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = np.argmax(output)
        if pred==y:
            correct += 1
         # 计算准确率
    accuracy = correct / len(test_loader.dataset) * 100
    print("Test Accuracy: {}%".format(accuracy))