using CuArrays, CUDAnative, CUDAdrv, LinearAlgebra, MPI

function copy(a, b)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y

    @inbounds b[x,y] = a[x,y]

    return nothing
end

function gpu_transpose(a, b)

    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y

    @inbounds b[y,x] = a[x,y]

    return nothing
end

function gpu_transpose_wblocks(a, b, res)

    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y

    xbound = Int(ceil(size(a,2)/res))
    ybound = Int(ceil(size(a,1)/res))
    for i = 1:xbound
        for j = 1:ybound
            if ((i-1)*res + x <= size(a,2) && (j-1)*res + y <= size(a,1))
                @inbounds b[(i-1)*res + x, (j-1)*res + y] = a[y + (j-1)*res, (i-1)*res + x]
                #@cuprintf("%ld\t%ld\t%ld\t%ld\n", (i-1)*res + x, (j-1)*res + y, y + (j-1)*res, (i-1)*res + x)
            end
        end
    end

    return nothing
end

function print_matrix(a)
    for i = 1:size(a,1)
        printstring = ""
        for j = 1:size(a,2)
            printstring *= string(a[i,j])*'\t'
        end
        println(printstring)
    end
end

function init(resx, resy)
    temp = zeros(resx, resy)
    for i = 1:resx
        temp[i,:] = [j for j=1:resy]
    end
    return (temp, zeros(resy, resx), CuArray(temp), CuArray(zeros(resy, resx)), res)

end

function GPU_transfer_test(comm)

    MPI.Barrier(comm)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    a = zeros(3)

    d_a = CuArray(zeros(3))

    if (rank == 0)
        a = [1.0, 2, 3]
        d_a = CuArray(a)
        a_check = Array(d_a)

        println(a_check == a, '\t', rank)

        sreq = MPI.Isend(d_a,1,1,comm)
        sreq = MPI.Isend(a,1,0,comm)
    end


    if (rank == 1)
        a_check = [1.0, 2.0, 3.0]
        b = Array{Float64}(undef, 3)

        rreq_gpu = MPI.Irecv!(d_a, 0, 1, comm)
        rreq_cpu = MPI.Irecv!(b, 0, 0, comm)
        MPI.Waitall!([rreq_cpu])
        MPI.Waitall!([rreq_gpu])

        println(a_check == Array(d_a), '\t', rank)
        println(a_check == b, '\t', rank)

    end

end

function dist_transpose(a, tile_size, comm)
    rank = MPI.Comm_rank(comm)
    mpisize = MPI.Comm_size(comm)

    tile = CuArray(zeros(tile_size, tile_size))
    c = similar(a)

    for r = 1:mpisize
        for i = 1:Int(ceil(size(a,2)/tile_size))
            @cuda threads = (tile_size, 1, 1) blocks = (1, tile_size, 1) gpu_transpose_wblocks(a[:,(i-1)*tile_size+1:i*tile_size], tile, tile_size)
            #println("i is: ",i, " and r is: ", r)
            if (i != rank+1)
                src = i-1
                tile_temp = similar(tile)
                rreq = MPI.Irecv!(tile_temp,src,src+r*10,comm)
                sreq = MPI.Isend(tile,src,rank+r*10,comm)
                MPI.Waitall!([rreq, sreq])
                tile = tile_temp
            end
            c[:,(i-1)*tile_size+1:i*tile_size] = tile
        end
    end

    println()
    for i = 0:mpisize
        if (rank == i)
            print_matrix(c)
        end
        MPI.Barrier(comm)
    end

    return a
end

function create_dist_matrix(resx, resy, comm)

    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    width = Int(ceil(resy/size))
    a = zeros(width, resx)
    for i = 1:resx
        for j = 1:width
            a[j,i] = float(j + rank*width)
        end
    end

    for i = 0:size
        if (rank == i)
            print_matrix(a)
        end
        MPI.Barrier(comm)
    end

    return CuArray(a)

end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD

#=
    resx = 3
    resy = 22
    #blocksize = min(resx, resy)
    blocksize = 3
    (a, b, d_a, d_b) = init(resx, resy)
    @cuda threads = (blocksize, 1, 1) blocks = (1, blocksize, 1) gpu_transpose_wblocks(d_a, d_b, blocksize)

    a = Array(d_a)
    b = Array(d_b)

    print_matrix(a)
    println()
    print_matrix(b)

    GPU_transfer_test(comm)
=#

    a = create_dist_matrix(12, 12,comm)
    b = dist_transpose(a, 4, comm)

    MPI.Finalize()
end

main()
