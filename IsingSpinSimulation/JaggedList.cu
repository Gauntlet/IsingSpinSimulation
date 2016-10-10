#include "DataStructures.h"
#include "JaggedList.h"
#include <numeric>

using namespace kspace;

template <class elem_type>
JaggedList<elem_type>::JaggedList(const uint32_t N, const uint32_t* lengths, const MemoryLocation memloc) : get(*this), set(*this)
{
	memloc = memloc;

	uint32_t* tmpoffsets = new uint32_t[ N + 1 ]();
	std::partial_sum( lengths, lengths + N, tmpoffsets + 1 );

	if ( memloc == MemoryLocation::host )
	{
		data = new elem_type[ tmpoffsets[ N ] ]();
		length = new uint32_t();
		size = new uint32_t();
		lengths = new uint32_t[ N ]();
		offsets = new uint32_t[ N + 1 ]();

		( *length ) = tmpoffsets[ N ];
		( *size ) = N;
		memcpy( lengths, lengths, sizeof( uint32_t ) * N );
		memcpy( offsets, tmpoffsets, sizeof( uint32_t ) * ( N + 1 ) );
	}
	else if ( memloc == MemoryLocation::device )
	{
		HANDLE_ERROR( cudaMalloc( (void**) &data, sizeof( elem_type ) * ( tmpoffsets[ N ] ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &length, sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &lengths, sizeof( uint32_t ) * N ) );
		HANDLE_ERROR( cudaMalloc( (void**) &offsets, sizeof( uint32_t ) * ( N + 1 ) ) );

		HANDLE_ERROR( cudaMemset( data, 0, sizeof( elem_type ) * tmpoffsets[ N ] ) );

		HANDLE_ERROR( cudaMemcpy( length, &N, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( lengths, lengths, sizeof( uint32_t ) * N, cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( offsets, tmpoffsets, sizeof( uint32_t ) * ( N + 1 ), cudaMemcpyHostToDevice ) );
	}

	delete[] tmpoffsets;
}

template <class elem_type>
JaggedList<elem_type>::~JaggedList()
{
	if ( memory_location() == MemoryLocation::host )
	{
		delete[] data;
		delete length;
		delete size;
		delete[] lengths;
		delete[] offsets;
	}
	else if ( memory_location() == MemoryLocation::device )
	{
		HANDLE_ERROR( cudaFree( data ) );
		HANDLE_ERROR( cudaFree( length ) );
		HANDLE_ERROR( cudaFree( size ) );
		HANDLE_ERROR( cudaFree( lengths ) );
		HANDLE_ERROR( cudaFree( offsets ) );
	}
}
