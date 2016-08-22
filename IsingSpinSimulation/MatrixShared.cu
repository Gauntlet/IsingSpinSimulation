#include "DataStructures.h"

using namespace kspace;

template <class elem_type>
MatrixShared::initialize( const uint32_t numofcols, const uint32_t numofrows )
{
	host = new Matrix( numofcols, numofrows, MemoryLocation::host );
	intermediary = new Matrix( numofcols, numofrows, MemoryLocation::device );
	HANDLE_ERROR( cudaMalloc((void**)&device, sizeof(Matrix)) );
	HANDLE_ERROR( cudaMemcpy(device, intermediary, sizeof(Matrix), cudaMemcpyHostToDevice) );
	

	HANDLE_ERROR( cudaMalloc(intermediary->_memloc, host->_memloc, sizeof(MemoryLocation), cudaMemcpyHostToDevice);
	HANDLE_ERROR( cudaMalloc(intermediary->_data, host->_data, sizeof(elem_type) * host->length(), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMalloc(intermediary->_length, host->_length, sizeof(uint32_t), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMalloc(intermediary->_numOfCols, host->_numOfCols, sizeof(uint32_t), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMalloc(intermediary->_numOfRows, host->_numOfRows, sizeof(uint32_t), cudaMemcpyHostToDevice) );
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
	HANDLE_ERROR( cudaFree(device) );
	delete[] intermediary;
	delete[] host;
}

template <class elem_type>
void MatrixShared::host2device()
{
	HANDLE_ERROR( cudaMalloc(intermediary->_data, host->_data, sizeof(elem_type) * host->length(), cudaMemcpyHostToDevice) );
}

template <class elem_type>
void MatrixShared::device2host()
{
	HANDLE_ERROR( cudaMalloc(host->_data, intermediary->_data, sizeof(elem_type) * host->length(), cudaMemcpyDeviceToHost) );
}