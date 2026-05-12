"""为 task096, task118, task266 构建修复模型"""
import onnx
import onnx.helper as helper
import onnx.numpy_helper as np_helper
import numpy as np

CH, H, W = 10, 30, 30

def make_identity():
    """1x1 Conv identity model - 保底方案"""
    # Weight: [out_channels, in_channels, kH, kW]
    # Identity: output[c] = input[c]
    w = np.zeros((CH, CH, 1, 1), dtype=np.float32)
    for c in range(CH):
        w[c, c, 0, 0] = 1.0
    
    w_tensor = np_helper.from_array(w, name='W')
    
    # Input/output
    x = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, CH, H, W])
    y = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, CH, H, W])
    
    # Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "W"],
        outputs=["output"],
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        name="identity_conv"
    )
    
    # Graph
    graph = helper.make_graph(
        [conv_node],
        "identity_model",
        [x],
        [y],
        initializer=[w_tensor]
    )
    
    # Model
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 11)])
    return model

# 修复三个任务
tasks = ['task096', 'task118', 'task266']
for task in tasks:
    model = make_identity()
    output_path = f'/Users/wxc/Documents/codes/neurogolf/submission/{task}.onnx'
    onnx.save(model, output_path)
    print(f'✅ {task}: 已写入恒等模型')
    
    # 验证
    try:
        onnx.checker.check_model(model, full_check=True)
        print(f'   ONNX checker passed')
    except Exception as e:
        print(f'   ONNX checker error: {e}')
