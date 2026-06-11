# 面向 ML 的 Julia 微积分. 数值 + 解析导数,
# 多元梯度, 梯度下降, Hessian 曲率,
# Taylor 展开, 以及用 SGD 训练的微小线性回归.
# 仅使用标准库. 来源:
#   https://docs.julialang.org/en/v1/manual/functions/
#   https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/
#   https://docs.julialang.org/en/v1/manual/arrays/

using Random
using LinearAlgebra
using Printf


function numerical_derivative(f, x::Float64; h::Float64=1e-7)::Float64
    return (f(x + h) - f(x - h)) / (2h)
end


function numerical_gradient(f, point::Vector{Float64}; h::Float64=1e-7)::Vector{Float64}
    n = length(point)
    grad = zeros(Float64, n)
    for i in 1:n
        plus = copy(point)
        minus = copy(point)
        plus[i] += h
        minus[i] -= h
        grad[i] = (f(plus) - f(minus)) / (2h)
    end
    return grad
end


function gradient_descent_1d(df, x0::Float64; lr::Float64=0.1, steps::Int=20)
    x = x0
    history = Tuple{Int, Float64, Float64}[]
    for step in 0:(steps - 1)
        g = df(x)
        x -= lr * g
        push!(history, (step, x, x * x))
    end
    return x, history
end


function hessian_2d(f, x::Float64, y::Float64; h::Float64=1e-5)
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h * h)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h * h)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h)
    return Float64[fxx fxy; fxy fyy]
end


function hessian_eigenvalues(H::Matrix{Float64})
    # Symmetric Hessian has real eigenvalues. Use stdlib eigvals via the LinearAlgebra dependency.
    return eigvals(Symmetric(H))
end


function taylor_approx(f, f_prime, f_double_prime, x0::Float64, h::Float64; order::Int=2)::Float64
    result = f(x0)
    if order >= 1
        result += f_prime(x0) * h
    end
    if order >= 2
        result += 0.5 * f_double_prime(x0) * h * h
    end
    return result
end


function demo_numerical_vs_analytical()
    println("=" ^ 55)
    println("数值 vs 解析导数")
    println("=" ^ 55)

    cases = [
        ("x^2", x -> x^2, x -> 2x),
        ("x^3", x -> x^3, x -> 3 * x^2),
        ("sin(x)", x -> sin(x), x -> cos(x)),
        ("e^x", x -> exp(x), x -> exp(x)),
        ("1/x", x -> 1 / x, x -> -1 / x^2),
    ]

    x = 2.0
    println("\n在 x = $x:")
    @printf("%-12s %12s %12s %12s\n", "函数", "数值", "解析", "误差")
    println("-" ^ 50)
    for (name, f, df) in cases
        num = numerical_derivative(f, x)
        ana = df(x)
        err = abs(num - ana)
        @printf("%-12s %12.6f %12.6f %12.2e\n", name, num, ana, err)
    end
end


function demo_gradient()
    println("\n" * "=" ^ 55)
    println("梯度（偏导数向量）")
    println("=" ^ 55)

    f = p -> p[1]^2 + 3 * p[1] * p[2] + p[2]^2

    point = Float64[1.0, 2.0]
    grad = numerical_gradient(f, point)
    analytical = Float64[2 * point[1] + 3 * point[2], 3 * point[1] + 2 * point[2]]

    println("\nf(x, y) = x^2 + 3xy + y^2")
    println("在点 ($(point[1]), $(point[2])):")
    @printf("  数值梯度:  [%.4f, %.4f]\n", grad[1], grad[2])
    @printf("  解析梯度: [%.1f, %.1f]\n", analytical[1], analytical[2])
end


function demo_gradient_descent_1d()
    println("\n" * "=" ^ 55)
    println("梯度下降: f(x) = x^2")
    println("=" ^ 55)

    x = 5.0
    lr = 0.1
    println("\n起始: x=$x, lr=$lr")
    for step in 0:19
        g = 2x
        x -= lr * g
        if step % 4 == 0 || step == 19
            @printf("  步 %2d  x=%8.4f  f(x)=%10.6f\n", step, x, x * x)
        end
    end
    @printf("在 x=%.6f 找到最小值 (真实最小值: x=0)\n", x)
end


function demo_gradient_descent_2d()
    println("\n" * "=" ^ 55)
    println("梯度下降: f(x, y) = x^2 + y^2")
    println("=" ^ 55)

    f = p -> p[1]^2 + p[2]^2
    point = Float64[4.0, 3.0]
    lr = 0.1
    @printf("\n起始: (%.1f, %.1f), lr=%.2f\n", point[1], point[2], lr)
    for step in 0:29
        g = numerical_gradient(f, point)
        point .-= lr .* g
        if step % 5 == 0 || step == 29
            @printf("  步 %2d  (%7.4f, %7.4f)  f=%.6f\n", step, point[1], point[2], f(point))
        end
    end
    @printf("在 (%.4f, %.4f) 找到最小值 (真实: (0, 0))\n", point[1], point[2])
end


function demo_hessian()
    println("\n" * "=" ^ 55)
    println("HESSIAN 矩阵: 鞍点 vs 最小值")
    println("=" ^ 55)

    saddle = (x, y) -> x^2 - y^2
    bowl = (x, y) -> x^2 + y^2
    rosenbrock = (x, y) -> (1 - x)^2 + 100 * (y - x^2)^2

    println("\nf(x, y) = x^2 - y^2 (鞍函数)")
    H = hessian_2d(saddle, 0.0, 0.0)
    evals = hessian_eigenvalues(H)
    println("  Hessian 在 (0, 0):")
    @printf("    [%6.2f  %6.2f]\n", H[1, 1], H[1, 2])
    @printf("    [%6.2f  %6.2f]\n", H[2, 1], H[2, 2])
    @printf("  特征值: %.2f, %.2f\n", evals[1], evals[2])
    println("  混合符号 => 鞍点")

    println("\nf(x, y) = x^2 + y^2 (碗函数)")
    H = hessian_2d(bowl, 0.0, 0.0)
    evals = hessian_eigenvalues(H)
    println("  Hessian 在 (0, 0):")
    @printf("    [%6.2f  %6.2f]\n", H[1, 1], H[1, 2])
    @printf("    [%6.2f  %6.2f]\n", H[2, 1], H[2, 2])
    @printf("  特征值: %.2f, %.2f\n", evals[1], evals[2])
    println("  均为正 => 局部最小值")

    println("\nRosenbrock f(x, y) = (1-x)^2 + 100(y - x^2)^2")
    H = hessian_2d(rosenbrock, 1.0, 1.0)
    evals = hessian_eigenvalues(H)
    println("  Hessian 在最小值 (1, 1):")
    @printf("    [%8.2f  %8.2f]\n", H[1, 1], H[1, 2])
    @printf("    [%8.2f  %8.2f]\n", H[2, 1], H[2, 2])
    @printf("  特征值: %.2f, %.2f\n", evals[1], evals[2])
    println("  均为正 => 局部最小值 (确认)")
end


function demo_taylor()
    println("\n" * "=" ^ 55)
    println("TAYLOR 级数近似")
    println("=" ^ 55)

    x0 = 1.0
    println("\n近似 f(x) = e^x 在 x0 = $x0 附近")
    @printf("%8s  %14s  %10s  %10s  %10s\n", "h", "真实 f(x0+h)", "0 阶", "1 阶", "2 阶")
    println("-" ^ 60)
    for h in [0.1, 0.5, 1.0, 2.0]
        true_val = exp(x0 + h)
        t0 = taylor_approx(exp, exp, exp, x0, h; order=0)
        t1 = taylor_approx(exp, exp, exp, x0, h; order=1)
        t2 = taylor_approx(exp, exp, exp, x0, h; order=2)
        @printf("%8.1f  %14.6f  %10.6f  %10.6f  %10.6f\n", h, true_val, t0, t1, t2)
    end

    println("\n近似 f(x) = sin(x) 在 x0 = 0 附近")
    @printf("%8s  %14s  %10s  %10s  %10s\n", "h", "真实 sin(h)", "0 阶", "1 阶", "2 阶")
    println("-" ^ 60)
    for h in [0.1, 0.5, 1.0, 2.0]
        true_val = sin(h)
        t0 = taylor_approx(sin, cos, x -> -sin(x), 0.0, h; order=0)
        t1 = taylor_approx(sin, cos, x -> -sin(x), 0.0, h; order=1)
        t2 = taylor_approx(sin, cos, x -> -sin(x), 0.0, h; order=2)
        @printf("%8.1f  %14.6f  %10.6f  %10.6f  %10.6f\n", h, true_val, t0, t1, t2)
    end

    println("\n关键洞察: 项越多 = 在 x0 附近近似越好,")
    println("但所有 Taylor 近似在远离 x0 时都会发散。")
end


function demo_linear_regression()
    println("\n" * "=" ^ 55)
    println("梯度下降: 线性回归 y = 2x + 1")
    println("=" ^ 55)

    Random.seed!(42)
    w = randn()
    b = randn()
    lr = 0.01

    xs = Float64[1, 2, 3, 4, 5]
    ys = Float64[3, 5, 7, 9, 11]
    n = length(xs)

    for epoch in 0:199
        total_loss = 0.0
        dw = 0.0
        db = 0.0
        for i in 1:n
            pred = w * xs[i] + b
            err = pred - ys[i]
            total_loss += err * err
            dw += 2 * err * xs[i]
            db += 2 * err
        end
        dw /= n
        db /= n
        total_loss /= n
        w -= lr * dw
        b -= lr * db
        if epoch % 40 == 0 || epoch == 199
            @printf("  轮 %3d  w=%.4f  b=%.4f  loss=%.6f\n", epoch, w, b, total_loss)
        end
    end

    @printf("\n学得: y = %.2fx + %.2f\n", w, b)
    println("实际:  y = 2.00x + 1.00")
end


function main()
    demo_numerical_vs_analytical()
    demo_gradient()
    demo_gradient_descent_1d()
    demo_gradient_descent_2d()
    demo_hessian()
    demo_taylor()
    demo_linear_regression()
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
