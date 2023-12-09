#include <iostream>
#include <vector>
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

double* approximations(const std::vector<int>& intervals) {
    int size = intervals.size();
    double* results = new double[size];

    for (int i = 0; i < size; ++i) {
        PiResults pi_results = pi_approx(intervals[i]);
        results[i] = pi_results.approx * 4; // Store the computed approximation for each interval
    }

    return results;
}

int main() {
    std::vector<int> intervals = {10, 20, 30}; // Sample intervals

    double* pi_approximations = approximations(intervals);

    // Display the results
    int size = intervals.size();
    for (int i = 0; i < size; ++i) {
        std::cout << "Approximation for intervals " << intervals[i] << ": " << pi_approximations[i] << std::endl;
    }

    // Deallocate the dynamically allocated array
    delete[] pi_approximations;

    return 0;
}
