#include <iostream>
#include <cmath>

struct PiResults {
    double approx;
    double error;
};

double f(double x) {
    return sqrt(1 - x * x); // Function to integrate
}

PiResults pi_approx(int N) {
    double a = 0.0; // Lower limit of integration
    double b = 1.0; // Upper limit of integration
    double h = (b - a) / N;
    double sum = 0.0;

    for (int k = 1; k < N; ++k) {
        double xk = a + k * h;
        sum += f(xk);
    }

    sum += (f(a) + f(b)) / 2.0; // Add the half values at a and b
    sum *= h;

    // Exact value of pi
    double exact_pi = M_PI;

    // Calculate absolute error
    double error = std::abs(exact_pi - (sum * 4)); // Multiply by 4 to get pi

    PiResults results;
    results.approx = sum; // Return the calculated sum as is
    results.error = error;

    return results;
}

int main() {
    int N;
    std::cout << "Enter the number of intervals (N): ";
    std::cin >> N;

    PiResults pi_results = pi_approx(N);

    std::cout << "Approximated value of pi: " << pi_results.approx * 4 << std::endl; // Multiply by 4 here
    std::cout << "Absolute error: " << pi_results.error << std::endl;

    return 0;
}