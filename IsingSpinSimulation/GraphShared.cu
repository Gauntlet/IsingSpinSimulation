#include "GraphShared.h"
#include <string>
using namespace kspace::GRAPH;

GraphShared::GraphShared(const std::string filename)
{
	host_ptr = new Graph( filename, MemoryLocation::host );
	intermediary_ptr = new Graph( filename, MemoryLocation::device );
	cudaMemcpy( device_ptr, intermediary_ptr, sizeof( Graph ), cudaMemcpyHostToDevice );
}

GraphShared::~GraphShared()
{
	if ( nullptr != device )
	{
		cudaFree( device_ptr );
	}

	if ( nullptr != intermediary_ptr )
	{
		delete intermediary_ptr;
	}

	if ( nullptr != host_ptr )
	{
		delete host_ptr;
	}
}

void GraphShared::host2device()
{
	const std::size_t N = host().get.number_of_nodes();
	const std::size_t M = host().get.offset( N );

	cudaMemcpy( intermediary().set.adjmat(),	host().get.adjmat(),	sizeof( std::uint8_t ) * N * N,	cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary().set.adjlist(),	host().get.adjlist(),	sizeof( std::int32_t ) * M,		cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary().set.degrees(),	host().get.degrees(),	sizeof( std::int32_t ) * N,		cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary().set.offsets(),	host().get.offsets(),	sizeof( uint32_t ) * ( N + 1 ),	cudaMemcpyHostToDevice );
}

void GraphShared::device2host()
{
	const std::size_t N = host().get.number_of_nodes();
	const std::size_t M = host().get.offset( N );

	cudaMemcpy( host().set.adjmat(),	intermediary().get.adjmat(),	sizeof( std::uint8_t ) * N * N,	cudaMemcpyDeviceToHost );
	cudaMemcpy( host().set.adjlist(),	intermediary().get.adjlist(),	sizeof( std::int32_t ) * M,		cudaMemcpyDeviceToHost );
	cudaMemcpy( host().set.degrees(),	intermediary().get.degrees(),	sizeof( std::int32_t ) * N,		cudaMemcpyDeviceToHost );
	cudaMemcpy( host().set.offsets(),	intermediary().get.offsets(),	sizeof( uint32_t ) * ( N + 1 ),	cudaMemcpyDeviceToHost );
}