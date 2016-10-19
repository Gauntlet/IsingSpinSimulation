#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cstdint>

namespace kspace
{

#ifdef __CUDA_ARCH__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

	static void HandleError( cudaError_t err, const char* file, int line )
	{
		if ( err != cudaSuccess )
		{
			printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
			exit( EXIT_FAILURE );
		}
	}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

	enum class MemoryLocation : std::uint32_t { host, device };

}

#endif