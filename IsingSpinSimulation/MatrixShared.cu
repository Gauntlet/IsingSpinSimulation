#include "DataStructures.h"

using namespace kspace;

template <class elem_type>
MatrixShared::initialize( const uint32_t numofcols, const uint32_t numofrows )
{
	host = new Matrix( numofcols, numofrows, MemoryLocation::host );
	intermediary = new Matrix( numofcols, numofrows, MemoryLocation::device );
	cudaMalloc( (void**) &device, sizeof( Matrix ) );
	cudaMemcpy( device, intermediary, sizeof( Matrix ), cudaMemcpyHostToDevice );
	

	cudaMalloc( intermediary->_memloc, host->_memloc, sizeof( MemoryLocation ), cudaMemcpyHostToDevice );
	cudaMalloc( intermediary->_data, host->_data, sizeof( elem_type ) * host->length(), cudaMemcpyHostToDevice );
	cudaMalloc( intermediary->_length, host->_length, sizeof( uint32_t ), cudaMemcpyHostToDevice );
	cudaMalloc( intermediary->_numOfCols, host->_numOfCols, sizeof( uint32_t ), cudaMemcpyHostToDevice );
	cudaMalloc( intermediary->_numOfRows, host->_numOfRows, sizeof( uint32_t ), cudaMemcpyHostToDevice );
}

template <class elem_type>
MatrixShared::MatrixShared( const uint32_t N )
{
	initialize( N, N );
}

template <class elem_type>
MatrixShared::MatrixShared( const uint32_t numofcols, const uint32_t numofrows )
{
	initialize( numofcols, numofrows );
}

template <class elem_type>
MatrixShared::~MatrixShared()
{
	cudaFree( device );
	delete[] intermediary;
	delete[] host;
}

template <class elem_type>
void MatrixShared::host2device()
{
	cudaMalloc( intermediary->_data,		host->_data,		sizeof( elem_type ) * host->length(),	cudaMemcpyHostToDevice );
}

template <class elem_type>
void MatrixShared::device2host()
{
	cudaMalloc( host->_data,		intermediary->_data,		sizeof( elem_type ) * host->length(),	cudaMemcpyDeviceToHost );
}