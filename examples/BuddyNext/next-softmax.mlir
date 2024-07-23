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

func.func @kenerl(%arg0 : tensor<1x32x40x128xf32>, %arg1 : tensor<1x32x40x128xf32>, %arg2 : tensor<1x1x40x40xf32>) {
    %t_start = call @rtclock() : () -> f64

    %90 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %91 = tosa.transpose %arg0, %90 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %92 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %93 = tosa.add %arg1, %92 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %94 = tosa.reshape %93 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %95 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %96 = tosa.add %91, %95 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %97 = tosa.reshape %96 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %98 = tosa.matmul %94, %97 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %99 = tosa.reshape %98 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %100 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %101 = tosa.reciprocal %100 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %102 = tosa.mul %99, %101 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %103 = tosa.add %102, %arg2 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %104 = tosa.reduce_max %103 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %105 = tosa.sub %103, %104 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %106 = tosa.exp %105 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %107 = tosa.reduce_sum %106 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %108 = tosa.reciprocal %107 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %109 = tosa.mul %106, %108 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64

    %tensor_unranked = tensor.cast %109 : tensor<1x32x40x40xf32> to tensor<*xf32>

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 32, 40, 40] strides = [51200, 1600, 40, 1] data = 
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [
    // CHECK-SAME: [0.025{{(, 0.025)*}}],

    // Print results.
    call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
    // Print timings.
    vector.print %time : f64

    return
}

func.func @main() {

    %X = arith.constant dense<-3.0> : tensor<1x32x40x128xf32>
    %c4 = arith.constant dense<4.0> : tensor<1x32x40x128xf32>
    %bias = arith.constant dense<7.0> : tensor<1x1x40x40xf32>

    call @kenerl(%X, %c4, %bias) : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>, tensor<1x1x40x40xf32>) -> ()

    return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)


