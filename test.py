import torch
import onnx
import onnxruntime
import numpy as np
def test(model, test_loader):
    """测试"""

    # 测试模式
    model.eval()

    # 存放正确个数
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:

            # GPU
            if torch.cuda.is_available():
                model = model.cuda()
                x, y = x.cuda(), y.cuda()
            # 获取结果
            output = model(x)
            # 预测结果
            pred = output.argmax(dim=1, keepdim=True)
            # 计算准确个数
            correct += pred.eq(y.view_as(pred)).sum().item()

    # 计算准确率
    accuracy = correct / len(test_loader.dataset) * 100

    # 输出准确
    print("Test Accuracy: {}%".format(accuracy))

if __name__=="__main__":
    model=onnx.load('./model.onnx')
    # 检查模型格式是否完整及正确
    onnx.checker.check_model(model)
    # 获取输出层，包含层名称、维度信息
    output = model.graph.output
    seesion=onnxruntime.InferenceSession('./model.onnx')
    x=np.random.randn(1,1,28,28).astype(np.float32)
    x=np.clip(x, 0, 1)
    input_name=seesion.get_inputs()[0].name
    output_name =seesion.get_outputs()[0].name
    pred=seesion.run([output_name],{input_name:x})
    pred=np.array(pred)
    print(pred)
    