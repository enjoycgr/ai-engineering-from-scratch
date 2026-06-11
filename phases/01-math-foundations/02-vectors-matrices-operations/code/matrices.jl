using LinearAlgebra


function demo_vectors()
    println("=" ^ 60)
    println("向量运算")
    println("=" ^ 60)

    v = [3.0, 4.0]
    w = [1.0, 2.0]

    println("\nv = $v")
    println("w = $w")
    println("v + w = $(v + w)")
    println("v - w = $(v - w)")
    println("v * 2 = $(v * 2)")
    println("v · w = $(dot(v, w))")
    println("|v| = $(norm(v))")
    println("v 归一化 = $(normalize(v))")
    println("|v 归一化| = $(norm(normalize(v)))")
end


function demo_basic_operations()
    println("\n" * "=" ^ 60)
    println("基本矩阵运算")
    println("=" ^ 60)

    A = [1 2; 3 4]
    B = [5 6; 7 8]

    println("\nA = $A")
    println("B = $B")
    println("A + B = $(A + B)")
    println("A - B = $(A - B)")
    println("A * 3 = $(A * 3)")
    println("A .* B (element-wise，逐元素) = $(A .* B)")
    println("A * B (matrix multiply，矩阵乘法) = $(A * B)")
    println("A' (transpose，转置) = $(A')")
end


function demo_determinant_inverse()
    println("\n" * "=" ^ 60)
    println("行列式与逆矩阵")
    println("=" ^ 60)

    A = [4 7; 2 6]
    println("\nA = $A")
    println("det(A) = $(det(A))")
    println("inv(A) = $(inv(A))")
    println("A * inv(A) = $(A * inv(A))")

    I3 = Matrix{Float64}(I, 3, 3)
    println("\n3x3 单位矩阵 = $I3")
end


function demo_broadcasting()
    println("\n" * "=" ^ 60)
    println("BROADCASTING (广播)")
    println("=" ^ 60)

    output = [1 2 3; 4 5 6]
    bias = [10 20 30]

    println("\nOutput = $output")
    println("Bias = $bias")
    println("Output .+ Bias = $(output .+ bias)")
end


function demo_neural_network_layer()
    println("\n" * "=" ^ 60)
    println("神经网络前向传播")
    println("=" ^ 60)

    input_size = 3
    hidden_size = 4
    output_size = 2

    x = [0.5, 0.8, 0.2]

    W1 = randn(hidden_size, input_size)
    b1 = zeros(hidden_size)
    W2 = randn(output_size, hidden_size)
    b2 = zeros(output_size)

    println("\n输入 x: $(size(x))")
    println("W1: $(size(W1))")
    println("W2: $(size(W2))")

    z1 = W1 * x .+ b1
    h1 = max.(0, z1)
    println("\n隐藏层 pre-activation z1 = $z1")
    println("隐藏层 post-ReLU h1 = $h1")

    z2 = W2 * h1 .+ b2
    println("输出 z2 = $z2")

    println("\n第 1 层: ($hidden_size x $input_size) * ($input_size,) -> ($hidden_size,)")
    println("第 2 层: ($output_size x $hidden_size) * ($hidden_size,) -> ($output_size,)")
end


function demo_weight_matrix_intuition()
    println("\n" * "=" ^ 60)
    println("权重矩阵直觉")
    println("=" ^ 60)

    W = [1.0 0.0 0.0;
         0.0 1.0 0.0;
         0.5 0.5 0.0]
    x = [0.8, 0.6, 0.1]

    println("\n权重矩阵 W:")
    display(W)
    println("\n\n输入 x = $x")
    println("W * x = $(W * x)")
    println("\n第 1 行: [1,0,0] 复制特征 1")
    println("第 2 行: [0,1,0] 复制特征 2")
    println("第 3 行: [0.5,0.5,0] 对特征 1 和 2 取平均")
end


function demo_julia_advantages()
    println("\n" * "=" ^ 60)
    println("JULIA 矩阵语法优势")
    println("=" ^ 60)

    A = [1 2; 3 4]
    println("\n矩阵字面量: A = [1 2; 3 4]")
    println("Transpose (转置): A' = $(A')")
    println("Matrix multiply (矩阵乘法): A * A = $(A * A)")
    println("Element-wise (逐元素): A .* A = $(A .* A)")
    println("Element-wise 函数: sin.(A) = $(sin.(A))")

    println("\nEigenvalues (特征值): $(eigvals(A))")
    println("Rank (秩): $(rank(A))")
    println("Trace (迹): $(tr(A))")

    println("\n矩阵除法 (求解 Ax = b):")
    b = [5.0, 11.0]
    x = A \ b
    println("A = $A, b = $b")
    println("x = A \\ b = $x")
    println("验证: A * x = $(A * x)")
end


demo_vectors()
demo_basic_operations()
demo_determinant_inverse()
demo_broadcasting()
demo_weight_matrix_intuition()
demo_julia_advantages()
demo_neural_network_layer()
