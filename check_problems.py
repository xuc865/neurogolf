"""检查有问题的模型的详细信息"""
import onnx
import sys

tasks = ['task096', 'task118', 'task266']
for task in tasks:
    path = f'submission/{task}.onnx'
    try:
        m = onnx.load(path)
        print(f"\n{'='*60}")
        print(f"{task}")
        print(f"{'='*60}")
        print(f"IR version: {m.ir_version}")
        print(f"Opset: {[ (o.domain, o.version) for o in m.opset_import ]}")
        print(f"Nodes ({len(m.graph.node)}):")
        for i, n in enumerate(m.graph.node):
            print(f"  {i:3d}. {n.op_type:20s} name={n.name or '(unnamed)'}")
            if n.output:
                print(f"       outputs: {list(n.output)}")
    except Exception as e:
        print(f"\n{task}: ERROR - {e}")
