#include <iostream>
#include <vector>
#include <cmath> // Include cmath for sqrt function and M_PI constant
using namespace std;

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
    double error = abs(exact_pi - (sum * 4)); // Multiply by 4 to get pi

    PiResults results;
    results.approx = sum; // Return the calculated sum as is
    results.error = error;

    return results;
}

double* approximations(const vector<int>& intervals) {
    int size = intervals.size();
    double* results = new double[size];

    for (int i = 0; i < size; ++i) {
        PiResults pi_results = pi_approx(intervals[i]);
        results[i] = pi_results.approx * 4; // Store the computed approximation for each interval
    }

    return results;
}

int main() {
    // Q1: Print approximation and error for N = 10000
    cout << "Q1 - Approximation for N = 10000:" << endl;
    PiResults pi_results = pi_approx(10000);
    cout << "Approximated value of pi: " << pi_results.approx * 4 << endl;
    cout << "Absolute error: " << pi_results.error << endl;

    // Q2: Create a vector with elements 101 to 107
    vector<int> intervals;
    for (int i = 101; i <= 107; ++i) {
        intervals.push_back(i);
    }

    // Q2: Print the elements from Q2 using the created vector
    cout << "\nQ2 - Approximations for intervals (101 to 107):" << endl;
    double* pi_approximations = approximations(intervals);
    for (size_t i = 0; i < intervals.size(); ++i) {
        cout << "Approximation for intervals " << intervals[i] << ": " << pi_approximations[i] << endl;
    }
    
    // Deallocate the dynamically allocated array
    delete[] pi_approximations;

    return 0;
}
