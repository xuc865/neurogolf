"""Fix SSA violations in task101.onnx - the root cause of Kaggle rejection."""
import onnx
from onnx import helper, TensorProto
from collections import Counter
import numpy as np

model_path = '/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx'
model = onnx.load(model_path)

# 1) Find all duplicate output tensor names
node_outputs = {}
seen_outputs = Counter()
for node in model.graph.node:
    for out in node.output:
        if out:
            seen_outputs[out] += 1

duplicate_outputs = {n for n, c in seen_outputs.items() if c > 1}
print(f'Duplicate output tensors: {len(duplicate_outputs)}')
for n in sorted(duplicate_outputs):
    count = seen_outputs[n]
    # Find which nodes produce this output
    producers = [node for node in model.graph.node if n in node.output]
    print(f'  {n}: produced by {len(producers)} nodes')

# 2) Fix by uniquifying output names
# Strategy: for each duplicate output, rename all but the last occurrence
# to a unique name, and update references

# Build a map of which nodes consume which tensors
consumers = {}
for node in model.graph.node:
    for inp in node.input:
        if inp not in consumers:
            consumers[inp] = []
        consumers[inp].append(node)

# Fix duplicates
replacements = {}  # old_name -> new_name
for dup_name in list(duplicate_outputs):
    # Find all producers in order
    producers = []
    for i, node in enumerate(model.graph.node):
        if dup_name in node.output:
            producers.append((i, node))
    
    print(f'\nFixing {dup_name}: {len(producers)} producers')
    
    # Keep the LAST occurrence, rename earlier ones
    for idx, (node_idx, node) in enumerate(producers[:-1]):
        new_name = f'{dup_name}_{idx}'
        # Rename this output
        output_list = list(node.output)
        out_idx = output_list.index(dup_name)
        output_list[out_idx] = new_name
        # Replace the node's output
        del node.output[out_idx]
        node.output.insert(out_idx, new_name)
        
        # Track replacement
        replacements[dup_name] = replacements.get(dup_name, [])
        replacements[dup_name].append((new_name, dup_name))  # replace dup_name with new_name
        # Actually this is wrong - we need to update specific references

print(f'\nTotal replacements: {len(replacements)}')

# Actually, a better approach: for each node that produces a duplicate output,
# rename that specific output to be unique, and update references throughout the graph

# Let me rebuild more carefully
model2 = onnx.load(model_path)

# Map each output tensor to its unique producer (last one wins)
# and rename all earlier producers
node_output_map = {}  # tensor_name -> list of (node_idx, output_idx)
for node_idx, node in enumerate(model2.graph.node):
    for out_idx, out_name in enumerate(node.output):
        if out_name:
            if out_name not in node_output_map:
                node_output_map[out_name] = []
            node_output_map[out_name].append((node_idx, out_idx))

# Create rename map
rename_map = {}  # old_unique_tensor_name -> new_name
tensor_name_counts = Counter()

for tensor_name, producers in node_output_map.items():
    if len(producers) > 1:
        print(f'\n{tensor_name}: {len(producers)} producers')
        # Rename all but the last producer
        for node_idx, out_idx in producers[:-1]:
            old_name = f'{tensor_name}_p{node_idx}'
            rename_map[(node_idx, out_idx, tensor_name)] = old_name
            print(f'  Node {node_idx}, output {out_idx}: {tensor_name} -> {old_name}')

# Apply renames to graph
for (node_idx, out_idx, old_tensor_name), new_name in rename_map.items():
    node = list(model2.graph.node)[node_idx]
    output_list = list(node.output)
    assert output_list[out_idx] == old_tensor_name
    output_list[out_idx] = new_name
    del node.output[out_idx]
    node.output.insert(out_idx, new_name)

# Update all references
for node in model2.graph.node:
    input_list = list(node.input)
    changed = False
    for i, inp_name in enumerate(input_list):
        for (src_node_idx, src_out_idx, old_name), new_name in rename_map.items():
            if inp_name == old_name:
                input_list[i] = new_name
                changed = True
                break  # Only one replacement per tensor
    if changed:
        del node.input[:]
        node.input.extend(input_list)

# Also need to update graph value_info and output
for vi in model2.graph.value_info:
    if vi.name in [old for _, _, old in rename_map.keys()]:
        # Find the new name
        for (_, _, old_name), new_name in rename_map.items():
            if vi.name == old_name:
                vi.name = new_name
                break

# Fix node names (duplicate empty names or duplicate custom names)
node_name_counts = Counter()
for node in model2.graph.node:
    node_name_counts[node.name] += 1

for name, count in node_name_counts.most_common(5):
    if count > 1:
        print(f'\nDuplicate node name "{name}": {count} occurrences')
        # Rename duplicates
        found = 0
        for node in model2.graph.node:
            if node.name == name:
                if found > 0:
                    node.name = f'{name}_{found}'
                found += 1

# Save fixed model
output_path = '/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx'
onnx.save(model2, output_path)
print(f'\nSaved fixed model to {output_path}')

# Verify
try:
    onnx.checker.check_model(model2, full_check=True)
    print('✅ ONNX checker passed!')
except Exception as e:
    print(f'❌ ONNX checker: {str(e)[:200]}')

# Try loading
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    print(f'✅ Loaded: {sess.get_inputs()[0].shape} -> {sess.get_outputs()[0].shape}')
except Exception as e:
    print(f'❌ Load: {str(e)[:200]}')
