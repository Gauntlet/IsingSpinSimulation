#include "Graph.h"

using namespace kspace::Graph;

GraphShared::GraphShared( GraphShared& other )
{
	( *this ).intermediary = other.intermediary;
	( *this ).host = other.host;
	( *this ).device = other.device;

	other.intermediary = nullptr;
	other.host = nullptr;
	other.device = nullptr;
}

GraphShared::GraphShared( const std::string filename )
{
	host = new Graph( filename, MemoryLocation::host );
	intermediary = new Graph( filename, MemoryLocation::device );

	cudaMalloc( (void**) &device, sizeof( Graph ) );
	cudaMemcpy( device, intermediary, sizeof( Graph ), cudaMemcpyHostToDevice );
}

GraphShared::~GraphShared()
{
	if ( nullptr != device )
	{
		cudaFree( device );
	}

	if ( nullptr != intermediary )
	{
		delete intermediary;
	}

	if ( nullptr != host )
	{
		delete host;
	}
}

GraphShared& GraphShared::operator=( GraphShared&& rhs )
{
	( *this ).intermediary = rhs.intermediary;
	( *this ).host = rhs.host;
	( *this ).device = rhs.device;

	rhs.intermediary = nullptr;
	rhs.host = nullptr;
	rhs.device = nullptr;
}

void GraphShared::host2device()
{
	const size_t N = host->number_of_nodes();
	const size_t M = host->_offsets[ N ];
	cudaMemcpy( intermediary->_adjmat, host->_adjmat, sizeof( std::uint8_t ) * N * N, cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary->_adjlist, host->_adjlist, sizeof( std::int32_t ) * M, cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary->_degrees, host->_degrees, sizeof( std::int32_t ) * N, cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary->_offsets, host->_offsets, sizeof( uint32_t ) * ( N + 1 ), cudaMemcpyHostToDevice );
}

void GraphShared::device2host()
{
	const size_t N = host->number_of_nodes();
	const size_t M = host->_offsets[ N ];
	cudaMemcpy( host->_adjmat, intermediary->_adjmat, sizeof( std::uint8_t ) * N * N, cudaMemcpyDeviceToHost );
	cudaMemcpy( host->_adjlist, intermediary->_adjlist, sizeof( std::int32_t ) * M, cudaMemcpyDeviceToHost );
	cudaMemcpy( host->_degrees, intermediary->_degrees, sizeof( std::int32_t ) * N, cudaMemcpyDeviceToHost );
	cudaMemcpy( host->_offsets, intermediary->_offsets, sizeof( uint32_t ) * ( N + 1 ), cudaMemcpyDeviceToHost );
}