import os
import numpy as np
import onnxruntime
import subprocess, traceback
import onnx
from collections import OrderedDict


def local_shell(cmd: str):
    try:
        a = subprocess.check_output([cmd], shell=True)
        res = a.decode("utf-8")
        return True
    except Exception as ex:
        traceback(ex)
        return False


def process_all_layer_onnx(model_path, input_dict):
    onnx_model = onnx.load_model(model_path)
    for node in onnx_model.graph.node:
        for output in node.output:
            onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, input_dict)
    outputs_name = [x.replace(".", "_").replace("::", "__") for x in outputs]
    ort_outs = OrderedDict(zip(outputs_name, ort_outs))
    return ort_outs


def onnx_run(model_path):
    """
    onnx 模拟推理
    :param model_save_path: 模型路径
    :return:
    """
    input0 = np.fromfile("resources/input_data/x_pass_3.raw", dtype=np.float32).reshape(
        [1, 1, 2, 64])
    input1 = np.fromfile("resources/input_data/onnx__Unsqueeze_188.raw",
                         dtype=np.float32).reshape([1, 1, 2, 32])
    input2 = np.fromfile("resources/input_data/onnx__Unsqueeze_191.raw",
                         dtype=np.float32).reshape([1, 1, 2, 32])
    input_dict = {"x_pass.3": input0, "onnx::Unsqueeze_188": input1, "onnx::Unsqueeze_191": input2}
    res = process_all_layer_onnx(model_path, input_dict)
    return res


def htp_run(so_path, res_dict):
    ##Qnn SDK 安装地址
    QNN_SDK_ROOT = "/home/suohong/THIRD-ENV/qnn/2.17.0.231124"
    res_dir = "resources/res_dir"
    shell_str = f"{QNN_SDK_ROOT}/bin/envsetup.sh;  {QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-net-run  --debug " \
                f" --backend {QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so " \
                f" --output_dir {res_dir} " \
                f" --input_list resources/input_data/input_list.txt" \
                f"  --model {so_path} "
    res = OrderedDict()
    if local_shell(shell_str):
        for ele in res_dict:
            f_name = ele + ".raw"
            file_name = os.path.join(res_dir, "Result_0", f_name)
            if os.path.exists(file_name):
                res_data = res_dict[ele]
                shape = res_data.shape
                res_data = np.fromfile(file_name, dtype=np.float32).reshape(shape)
                res[ele] = res_data
        return res
    return None


def main():
    x_pass_3 = np.fromfile("resources/input_data/x_pass_3.raw", dtype=np.float32).reshape(
        [1, 1, 2, 64])
    Unsqueeze_188 = np.fromfile("resources/input_data/onnx__Unsqueeze_188.raw",
                         dtype=np.float32).reshape([1, 1, 2, 32])
    Unsqueeze_191 = np.fromfile("resources/input_data/onnx__Unsqueeze_191.raw",
                         dtype=np.float32).reshape([1, 1, 2, 32])
    input_dict = {"x_pass.3": x_pass_3, "onnx::Unsqueeze_188": Unsqueeze_188, "onnx::Unsqueeze_191": Unsqueeze_191}
    onnx_res = onnx_run("resources/sub_concat_1.onnx")
    htp_res = htp_run("resources/x86_64-linux-clang/libsub_concat_1.so",      onnx_res)
    # 取【0,0,0]纬度数据做对比
    diff_res_1 = np.array([onnx_res['onnx__Concat_205'][0, 0, 0], htp_res['onnx__Concat_205'][0, 0, 0]])

    htp_res_modify = htp_run(
        "resources/x86_64-linux-clang/libsub_concat_1_res.so",
        onnx_res)
    diff_res_2 = np.array([onnx_res['onnx__Concat_205'][0, 0, 0], htp_res_modify['onnx__Concat_205'][0, 0, 0]])
    print("---over---")


if __name__ == "__main__":
    main()
