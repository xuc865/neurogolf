"""Clean up task101.onnx: remove duplicate initializers and fix graph inputs."""
import onnx
from onnx import helper, TensorProto
import numpy as np

model_path = '/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx'
model = onnx.load(model_path)

# Check graph inputs
print("=== Graph inputs ===")
input_names = set()
for inp in model.graph.input:
    print(f"  {inp.name}: {inp.type.tensor_type.shape}")
    input_names.add(inp.name)

# Check graph outputs
print("\n=== Graph outputs ===")
for out in model.graph.output:
    print(f"  {out.name}")

# Find initializers that also appear as inputs (these should be removed from inputs)
initializer_names = {init.name for init in model.graph.initializer}
print(f"\n=== Initializer names ({len(initializer_names)}) ===")

# Find duplicates
from collections import Counter
init_names_list = [init.name for init in model.graph.initializer]
dups = [name for name, count in Counter(init_names_list).items() if count > 1]
if dups:
    print(f"\nDuplicate initializers: {dups}")

# Find inputs that are also initializers
inputs_that_are_init = input_names & initializer_names
if inputs_that_are_init:
    print(f"\nInputs that are also initializers (need to remove from inputs): {inputs_that_are_init}")

# Check for value_info (these should include intermediate tensors)
value_info_names = {vi.name for vi in model.graph.value_info}
print(f"\n=== Value info: {len(value_info_names)} tensors ===")

# Now let's create a cleaned model
# Step 1: Move initializers out of graph inputs
new_inputs = []
inputs_removed = 0
for inp in model.graph.input:
    if inp.name in initializer_names:
        inputs_removed += 1
        print(f"  Removing input '{inp.name}' (it's an initializer)")
    else:
        new_inputs.append(inp)

# Step 2: Remove duplicate initializers (keep only the last occurrence)
seen = {}
new_initializers = []
for init in reversed(model.graph.initializer):
    if init.name not in seen:
        seen[init.name] = True
        new_initializers.append(init)
new_initializers.reverse()

# Step 3: Rebuild the graph
new_graph = helper.make_graph(
    nodes=list(model.graph.node),
    name=model.graph.name,
    inputs=new_inputs,
    outputs=list(model.graph.output),
    initializer=new_initializers,
    value_info=list(model.graph.value_info),
    doc_string=model.graph.doc_string
)

# Rebuild the model
new_model = helper.make_model(new_graph, ir_version=model.ir_version,
                               producer_name=model.producer_name,
                               producer_version=model.producer_version,
                               domain=model.domain,
                               model_version=model.model_version,
                               doc_string=model.doc_string)
new_model.opset_import.extend(model.opset_import)

# Validate
print(f"\nOriginal inputs: {len(model.graph.input)}, New inputs: {len(new_graph.input)}")
print(f"Original initializers: {len(model.graph.initializer)}, New initializers: {len(new_graph.initializer)}")
print(f"Nodes unchanged: {len(model.graph.node)}")

# Save
output_path = '/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx'
onnx.save(new_model, output_path)
print(f"\nSaved cleaned model to {output_path}")

# Verify
model2 = onnx.load(output_path)
print(f"Verified: {len(model2.graph.input)} inputs, {len(model2.graph.initializer)} initializers")
