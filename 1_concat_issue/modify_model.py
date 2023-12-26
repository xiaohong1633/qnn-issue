import os
import numpy as np
import onnx
import onnxruntime
import onnxsim
import onnx_graphsurgeon as gs
from onnxsim import onnx_simplifier
from onnx import helper, shape_inference


def replace_concat_node(node):
    item_inputs = node.inputs
    item_outputs = node.outputs
    output_node = item_outputs[0]
    output_shape = output_node.shape
    assert len(item_inputs) == 2
    input0, input1 = item_inputs[0], item_inputs[1]
    shape0 = input0.shape
    shape1 = input1.shape
    axis = node.attrs["axis"]
    ##第一：concat节点
    concat_0_constant = gs.Constant(name=f"myself_concat_0_constant_{node.name}",
                                    values=np.zeros(shape=shape1, dtype=np.float32))
    concat_0_res_variable = gs.Variable(name=f"myself_variable_0_constant_{node.name}", shape=output_shape,
                                        dtype=np.float32)
    concat_node0 = gs.Node("Concat", name=f"myself_concat_add_{node.name}_0", inputs=[input0, concat_0_constant],
                           outputs=[concat_0_res_variable], attrs={"axis": axis})

    ##第二：concat节点
    concat_1_constant = gs.Constant(name=f"myself_concat_1_constant_{node.name}",
                                    values=np.zeros(shape=shape0, dtype=np.float32))
    concat_1_res_variable = gs.Variable(name=f"myself_variable_1_constant_{node.name}", shape=output_shape,
                                        dtype=np.float32)
    concat_node1 = gs.Node("Concat", name=f"myself_concat_add_{node.name}_1", inputs=[concat_1_constant, input1],
                           outputs=[concat_1_res_variable], attrs={"axis": axis})

    ##第三：Add节点
    # add_1_res_variable = gs.Variable(name=f"myself_variable_1_add_{node.name}", shape=output_shape,
    #                                  dtype=np.float32)
    add_node0 = gs.Node("Add", name=f"myself_add_{node.name}_0",
                        inputs=[concat_0_res_variable, concat_1_res_variable],
                        outputs=[output_node])
    node.inputs.clear()
    node.outputs.clear()
    return [add_node0, concat_node1, concat_node0]
def replace_concat_to_add(gs_graph):
    # 逻辑，替换concat两个输入，为先和同型0tensor concat，然后相加
    inputs = gs_graph.inputs
    outputs = gs_graph.outputs
    nodes = gs_graph.nodes
    tensors = gs_graph.tensors()

    concat_nodes = []
    for i, node in enumerate(nodes):
        if node.op == "Concat":
            print(node.name)
            concat_nodes.append((i, node))

    # 从尾到头逐一替换
    for item in concat_nodes[::-1]:
        idx, node = item
        last_node, middle_node, first_node = replace_concat_node(node)
        nodes.insert(idx, last_node)
        nodes.insert(idx, middle_node)
        nodes.insert(idx, first_node)

    gs_graph.cleanup()
    return gs_graph
def main():
    model_path = "qnn-issue/1_concat_issue/resources/sub_concat_1.onnx"
    onnx_model = onnx.load_model(model_path)
    gs_graph = gs.import_onnx(onnx_model)
    tensors = gs_graph.tensors()
    for item in tensors:
        if item=="onnx::Slice_75":
            tensor = tensors[item]
            tensor.__setattr__("values", np.array([4], dtype=np.int64))
            print(item)
    gs_graph = replace_concat_to_add(gs_graph)

    onnx_model = gs.export_onnx(gs_graph)
    simplified_model, _ = onnx_simplifier.simplify(onnx_model)
    # onnx.checker.check_model(model)
    # onnx.checker.check_model(simplified_model)
    onnx_model = shape_inference.infer_shapes(simplified_model)
    onnx.save_model(onnx_model, "qnn-issue/1_concat_issue/resources/sub_concat_1_res.onnx")
    print("---over---")

if __name__ == "__main__":
    main()