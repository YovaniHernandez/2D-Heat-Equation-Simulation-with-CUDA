# 2D-Heat-Equation-Simulation-with-CUDA
A high-performance numerical simulation of the 2D heat equation using the FTCS (Forward-Time Central-Space) finite difference method. This project compares a sequential CPU implementation against an optimized parallel GPU version using NVIDIA CUDA.

This project solves the heat diffusion equation on a 2D grid. It demonstrates the massive speedup capabilities of GPUs when handling stencil-based computations.

 Key FeaturesCUDA Acceleration: Parallel implementation using a 5-point stencil.Shared Memory Optimization: [Inferencia] Uses GPU shared memory and halo cells to minimize global memory latency.Double Buffering: Prevents race conditions during temporal iterations.Visualization: Python scripts to generate heatmaps and animations from simulation data.Performance Metrics: Automatic calculation of Speedup ($S = T_{CPU} / T_{GPU}$) and error validation.

The simulation uses the explicit FTCS scheme:Stability Condition: [No verificado] $r = \frac{\alpha \Delta t}{h^2} \le 0.25$Boundary Conditions: Constant Dirichlet ($T=10$).Initial State: Uniform temperature with a high-heat circular source in the center.

  Prerequisites
NVIDIA GPU (Compute Capability 3.0+)
CUDA Toolkit
GCC/G++ compiler
Python 3 (for visualization)

# Clone the repository
git clone https://github.com/youruser/cuda-heat-equation.git

# Compile the CUDA and CPU versions
nvcc -o heat_sim main.cu

# Run the simulation
./heat_sim

# Generate plots
python3 visualize_results.py
