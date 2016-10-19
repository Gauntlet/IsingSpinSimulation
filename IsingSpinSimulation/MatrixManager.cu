#include "MatrixManager.h"

#include "cuda_runtime.h"

using namespace kspace;

template <class elem_type>
void MatrixManager<elem_type>::initialise_host( const uint32_t number_of_columns, const uint32_t number_of_rows )
{
	host_ptr = new Matrix( number_of_columns, number_of_rows, MemoryLocation::host );
}

template <class elem_type>
void MatrixManager<elem_type>::initialise_intermediary( const uint32_t number_of_columns, const uint32_t number_of_rows )
{
	intermediary_ptr = new Matrix<elem_type>( number_of_columns, number_of_rows, MemoryLocation::device );

	//Copy the matrix data from host to device.
	HANDLE_ERROR( cudaMalloc( intermediary().data_ptr, host().get.data_ptr(), sizeof( elem_type ) * host().get.length(), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( intermediary().length, host().get.length(), sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( intermediary().number_of_columns, host().get.number_of_columns(), sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( intermediary().number_of_rows, host().get.number_of_rows(), sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
}

template <class elem_type>
void MatrixManager<elem_type>::initialise_device()
{
	HANDLE_ERROR( cudaMalloc( (void**) &device_ptr, sizeof( Matrix<elem_type> ) ) );

	//Copy the matrix data device pointers from the intermediary matrix to the device matrix.
	//Note now the device pointers are accessible from the intermediary matrix object and the device matrix object.
	HANDLE_ERROR( cudaMemcpy( device_ptr, intermediary_ptr, sizeof( Matrix<elem_type> ), cudaMemcpyHostToDevice ) );
}

template <class elem_type>
void MatrixManager<elem_type>::move_data( MatrixManager<elem_type>&& that )
{
	clear();

	intermediary_ptr = that.intermediary_ptr;
	host_ptr = that.host_ptr;
	device_ptr = that.device_ptr;

	that.intermediary_ptr = nullptr;
	that.host_ptr = nullptr;
	that.device_ptr = nullptr;
}

template <class elem_type>
void MatrixManager<elem_type>::move_data( Matrix<elem_type>&& that )
{
	clear();

	if ( MemoryLocation::host == that.get.memory_location() )
	{
		host_ptr = new Matrix<elem_type>( std::move(that) );
		initialise_intermediary( host().get.number_of_columns(), host().get.number_of_rows() );
		initialise_device();
	}
	else if ( MemoryLocation::device == that.get.memory_location() )
	{
		intermediary_ptr = new Matrix<elem_type>( std::move( that ) );

		std::uint32_t NoC, NoR;
		cudaMemcpy( &NoC, that.number_of_columns, sizeof( std::uint32_t ), cudaMemcpyDeviceToHost );
		cudaMemcpy( &NoR, that.number_of_rows, sizeof( std::uint32_t ), cudaMemcpyDeviceToHost );

		initialise_host( NoC, NoR );
		device2host();
		initialise_device();
	}
}

template <class elem_type>
void MatrixManager<elem_type>::clear()
{
	if ( nullptr != device_ptr )
	{
		HANDLE_ERROR( cudaFree( device_ptr ) );
	}

	if ( nullptr != intermediary_ptr )
	{
		delete[] intermediary;
	}

	if ( nullptr != host_ptr )
	{
		delete[] host_ptr;
	}
}

template <class elem_type>
MatrixManager<elem_type>::MatrixManager(const std::uint32_t N)
{
	initialise_host( N, N );
	initialise_intermediary( N, N );
	initialise_device();
}

template <class elem_type>
MatrixManager<elem_type>::MatrixManager( const uint32_t number_of_columns, const uint32_t number_of_rows )
{
	initialise_host( number_of_columns, number_of_rows );
	initialise_intermediary( number_of_columns, number_of_rows );
	initialise_device();
}

template <class elem_type>
MatrixManager::~MatrixManager()
{
	clear();
}

template<class elem_type>
MatrixManager<elem_type>::MatrixManager( Matrix<elem_type>&& that )
{
	move_data( that );
}

template<class elem_type>
MatrixManager<elem_type>& MatrixManager<elem_type>::operator=( Matrix<elem_type>&& that )
{
	move_data( that );
	return *this;
}

template <class elem_type>
void MatrixManager<elem_type>::host2device()
{
	HANDLE_ERROR( cudaMalloc( intermediary().data_ptr, host().get.data_ptr(), sizeof( elem_type ) * host().get.length(), cudaMemcpyHostToDevice ) );
}

template <class elem_type>
void MatrixManager<elem_type>::device2host()
{
	HANDLE_ERROR( cudaMalloc( host().data_ptr, intermediary().get.data_ptr(), sizeof( elem_type ) * host().get.length(), cudaMemcpyDeviceToHost ) );
}