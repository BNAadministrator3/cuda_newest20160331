#include "BC_GPU.cuh"

// changeable for performance 
#define VIRTUAL_WARP 16

__global__ void betweenness_node_succ_kernel(float *BC, int *r, int *c, int * dist, float * sigma, float * delta, int * S, int * S_end, int numVertices, int numEdges, int offset_source) 
{
	int offset_vertices = blockIdx.x * numVertices;
	int offset_edge = blockIdx.x * numEdges;

	for (int i = threadIdx.x; i < numVertices; i += blockDim.x)
	{
		dist[offset_vertices + i] = -1;
		sigma[offset_vertices + i] = 0;
		delta[offset_vertices + i] = 0;

		S_end[offset_vertices + i] = r[i];
	}

	for (int i = threadIdx.x; i < numEdges; i += blockDim.x)
	{
		S[offset_edge + i] = 0;
	}

	int edge_index = threadIdx.x % VIRTUAL_WARP;
	int vertice_index = threadIdx.x / VIRTUAL_WARP;
	int source = blockIdx.x + offset_source;
	if (source >= numVertices)
		return;
	__shared__ bool done;
	done = false;
	
	int level = 0;
	dist[offset_vertices + source] = level++;
	sigma[offset_vertices + source ] = 1; 

	while (!done)
	{
		__syncthreads(); // attention: this sync is neccessary
		done = true;
		for (int current = vertice_index; current < numVertices; current += blockDim.x / VIRTUAL_WARP)
		{
			if (dist[offset_vertices + current] != level - 1)
				continue;
			for (int j = r[current] + edge_index; j < r[current + 1]; j += VIRTUAL_WARP)
			{
				int next = c[j];
				int read_dist = dist[offset_vertices + next];
				if (read_dist == -1)
				{
					dist[offset_vertices + next] = level;
					done = false;
				}
				if (read_dist < level && read_dist >= 0)
					continue;

				atomicAdd(sigma + offset_vertices + next, sigma[offset_vertices + current]); //atomic!

				int s = atomicAdd(S_end + offset_vertices + current, 1);
				S[offset_edge + s] = next;
			}
		}
		level ++;
		__syncthreads();
	}

	for (int i = level - 2; i >= 0; i--)
	{
		for (int current = threadIdx.x; current < numVertices; current += blockDim.x)
		{
			if (dist[offset_vertices + current] != i)
				continue;
			for (int j = r[current]; j < S_end[offset_vertices + current]; j += 1)
			{
				int next = S[offset_edge + j];
				delta[offset_vertices + current] += (double) sigma[offset_vertices + current] / sigma[offset_vertices + next]*(1 + delta[offset_vertices + next]);
			}
		}
		__syncthreads();
	}

	for (int current = threadIdx.x; current < numVertices; current += blockDim.x)
	{
		if(current != source)
			atomicAdd(BC + current, delta[offset_vertices + current]);
	}
}

void Betweenness_GPU_node_succ(int *r, int *c, int numVertices, int numEdges, float *BC, int grid, int thread)
{
    int devID;
    cudaDeviceProp deviceProps;
    devID = findCudaDevice();
    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

	//int thread = 256;
	//int grid = 100;
    // allocate device memory
    int* d_r; 
	int* d_c;
	int* dist;
	float* sigma;
	float* delta;
	int* P;
	int* P_end;

    checkCudaErrors( cudaMalloc( (void**) &d_r, sizeof(int) * (numVertices + 1)));
	checkCudaErrors( cudaMalloc( (void**) &d_c, sizeof(int) * numEdges));
	checkCudaErrors( cudaMalloc( (void**) &dist, sizeof(int) * numVertices * grid));
	checkCudaErrors( cudaMalloc( (void**) &sigma, sizeof(int) * numVertices * grid));
	checkCudaErrors( cudaMalloc( (void**) &delta, sizeof(int) * numVertices * grid));
	checkCudaErrors( cudaMalloc( (void**) &P, sizeof(int) * numEdges * grid));
	checkCudaErrors( cudaMalloc( (void**) &P_end, sizeof(int) * numVertices * grid));

    // copy host memory to device
	checkCudaErrors( cudaMemcpy( d_r, r, sizeof(int) * (numVertices + 1), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy( d_c, c, sizeof(int) * numEdges, cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_BC;
    checkCudaErrors( cudaMalloc( (void**) &d_BC, sizeof(float) * numVertices));
	checkCudaErrors( cudaMemset( d_BC, 0, sizeof(float) * numVertices));

	// execute the kernel
	clock_t kernel_time = 0;
	for (int offset_source = 0; offset_source < numVertices; offset_source += grid)
	{
		clock_t time = clock();
		betweenness_node_succ_kernel<<<grid, thread>>>(d_BC, d_r, d_c, dist, sigma, delta, P, P_end, numVertices, numEdges, offset_source);
		// check if kernel execution generated and error
		getLastCudaError("Kernel execution failed");
		cudaThreadSynchronize();
		time = clock() - time;

		kernel_time += time;
		//cout<<offset_source<<" done. Time = "<<time<<"ms."<<endl;
	}
	cout<<"total kernel time: "<<kernel_time<<"ms."<<endl;

	// copy result from device to host
	checkCudaErrors(cudaMemcpy(BC, d_BC, sizeof(float) * numVertices, cudaMemcpyDeviceToHost));

    // cleanup memory
    checkCudaErrors(cudaFree(d_r));
    checkCudaErrors(cudaFree(d_c));
	checkCudaErrors(cudaFree(d_BC));
	checkCudaErrors(cudaFree(dist));
    checkCudaErrors(cudaFree(sigma));
	checkCudaErrors(cudaFree(delta));
	checkCudaErrors(cudaFree(P));
	checkCudaErrors(cudaFree(P_end));
    cudaDeviceReset();
}
