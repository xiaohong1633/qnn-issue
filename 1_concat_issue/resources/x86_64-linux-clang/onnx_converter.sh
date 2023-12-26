###模型转化脚本
cd /data01/suohong/sub_graph

name=sub_concat_1

${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter -i ./${name}.onnx  --param_quantizer symmetric  --input_list ./input_data/input_list.txt --act_bw 16 --bias_bw 32  -o ./output/${name}.cpp     --no_simplification
## --input_layout onnx::Unsqueeze_188  NONTRIVIAL --input_layout onnx::Unsqueeze_191  NONTRIVIAL --input_layout x_pass.3  NONTRIVIAL

$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator -c ./output/${name}.cpp  -l ${name}  -t x86_64-linux-clang -o ./output/
##修改模型后bin文件是必要参数
##$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator -c ./output/${name}.cpp -b ./output/${name}.bin -l $name  -t x86_64-linux-clang -o ./output/


$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator --output_dir ./output/context-binary --model ./output/x86_64-linux-clang/lib${name}.so --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so --binary_file ${name}_context


echo "---over---"