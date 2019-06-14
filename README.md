# DistributedTranspose
This repo is intended to scale n-dimensional tranposes to massively parallel systems, such as interconnected GPU devices or computer nodes found in high-performance computing

## Features to be implemented:
1. Non-Cuda aware MPI GPU transposes
2. Distributed CPU transposes (DistributedArrays.jl? GPUifyloops.jl?)
3. Optimized Matrix Transpose (https://www.cs.colostate.edu/~cs675/MatrixTranspose.pdf_
4. In-place transposes (https://www.researchgate.net/publication/221309329_An_efficient_in-place_3D_transpose_for_multicore_processors_with_software_managed_memory_hierarchy)

