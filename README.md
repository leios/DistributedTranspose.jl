# DistributedTranspose
This repo is intended to scale n-dimensional tranposes to massively parallel systems, such as interconnected GPU devices or computer nodes found in high-performance computing

## Features to be implemented:
1. Non-Cuda aware MPI GPU transposes
2. Distributed CPU transposes (DistributedArrays.jl? GPUifyloops.jl?)
3. Optimized Matrix Transpose (https://www.cs.colostate.edu/~cs675/MatrixTranspose.pdf_
4. In-place transposes (https://www.researchgate.net/publication/221309329_An_efficient_in-place_3D_transpose_for_multicore_processors_with_software_managed_memory_hierarchy)
5. GPU-GPU data transfer mediated by CPU (note when CUDA-aware MPI is enabled, use GPU-GPU)
6. Scale test to see what type of transpose (CPU / GPU) is best for your hardware

## ERROR
- Initialization of non-square matrices incorrect (outputs square when it should not)
- Tranpose on non-square matrices fails because:
    1. Tile used for transpose is square
    2. 2d indexing should be switched to 1d
    3. May be a problem with the MPI transfer
