// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -tensor-bufferize \
// RUN:     -buffer-deallocation \
// RUN:     -finalizing-bufferize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map7 = affine_map<(d0, d1) -> (0, d0, d1)>

func.func @kenerl(%arg0: tensor<1x40x4096xf32>, %arg1: tensor<1x40x11008xf32>, %arg3: tensor<11008x4096xf32>, %arg4: tensor<4096x11008xf32>) {
    %t_start = call @rtclock() : () -> f64

    %147 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %148 = tosa.transpose %arg3, %147 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %149 = tosa.reshape %arg0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_25 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%149, %148 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_25 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %151 = tosa.reshape %150 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %152 = tosa.mul %arg1, %151 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %153 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %154 = tosa.transpose %arg4, %153 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %155 = tosa.reshape %152 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_26 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %156 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%155, %154 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_26 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %157 = tosa.reshape %156 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    // %158 = tosa.add %arg0, %157 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>


    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64

    %tensor_unranked = tensor.cast %157 : tensor<1x40x4096xf32> to tensor<*xf32>

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data = 
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [4.43243e+07{{(, 4.43243e+07)*}}],

    // Print results.
    call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
    // Print timings.
    vector.print %time : f64

    return
}

func.func @main() {
    %c05 = arith.constant dense<0.5> : tensor<1x40x4096xf32>
    %c1 = arith.constant dense<1.0> : tensor<1x40x11008xf32>
    %f2 = arith.constant dense<0.2> : tensor<11008x4096xf32>
    %f3 = arith.constant dense<0.3> : tensor<4096x11008xf32>

    call @kenerl(%c05, %c1, %f2, %f3) : (tensor<1x40x4096xf32>, tensor<1x40x11008xf32>, tensor<11008x4096xf32>, tensor<4096x11008xf32>) -> ()

    return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)




