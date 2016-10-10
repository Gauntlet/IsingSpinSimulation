#include "DataStructures.h"
#include "MatrixShared.h"

using namespace kspace;

template <class elem_type>
void MatrixShared<elem_type>::initialise( const uint32_t number_of_columns, const uint32_t number_of_rows )
{
	host = new Matrix( number_of_columns, number_of_rows, MemoryLocation::host );
	intermediary = new Matrix( number_of_columns, number_of_rows, MemoryLocation::device );
	HANDLE_ERROR( cudaMalloc( (void**) &device, sizeof( Matrix ) ) );
	HANDLE_ERROR( cudaMemcpy( device, intermediary, sizeof( Matrix ), cudaMemcpyHostToDevice ) );


	HANDLE_ERROR( cudaMalloc( intermediary->_memloc, host->_memloc, sizeof( MemoryLocation ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( intermediary->_data, host->_data, sizeof( elem_type ) * host->length(), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( intermediary->_length, host->_length, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( intermediary->_numOfCols, host->_numOfCols, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( intermediary->_numOfRows, host->_numOfRows, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
}


template <class elem_type>
MatrixShared<elem_type>::MatrixShared( const std::uint32_t N )
{
	initialize( N, N );
}

template <class elem_type>
MatrixShared<elem_type>::MatrixShared( const uint32_t number_of_columns, const uint32_t number_of_rows )
{
	initialize( number_of_columns, number_of_rows );
}

template <class elem_type>
void MatrixShared<elem_type>::move_data( MatrixShared<elem_type>&& that )
{
	intermediary_ptr = that.intermediary_ptr;
	host_ptr = that.host_ptr;
	device_ptr = that.device_ptr;

	that.intermediary_ptr = nullptr;
	that.host_ptr = nullptr;
	that.device_ptr = nullptr;
}

template <class elem_type>
MatrixShared::~MatrixShared()
{
	HANDLE_ERROR( cudaFree( device ) );
	delete[] intermediary;
	delete[] host;
}

template <class elem_type>
void MatrixShared<elem_type>::host2device()
{
	HANDLE_ERROR( cudaMalloc( intermediary->_data, host->_data, sizeof( elem_type ) * host().length(), cudaMemcpyHostToDevice ) );
}

template <class elem_type>
void MatrixShared<elem_type>::device2host()
{
	HANDLE_ERROR( cudaMalloc( host->_data, intermediary->_data, sizeof( elem_type ) * host->length(), cudaMemcpyDeviceToHost ) );
}