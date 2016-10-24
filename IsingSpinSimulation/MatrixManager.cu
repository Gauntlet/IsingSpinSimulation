#include "MatrixManager.h"
#include "cuda_runtime.h"

using namespace kspace;

/**
* Creates a number_of_columns x number_of_rows Matrix on the host with the data stored on the host.
* @param number_of_columns an integer.
* @param number_of_rows an integer.
*/
template <class elem_type>
void MatrixManager<elem_type>::initialise_host( const uint32_t number_of_columns, const uint32_t number_of_rows )
{
	host_ptr = new Matrix( number_of_columns, number_of_rows, MemoryLocation::host );
}

/**
* Creates a number_of_columns x number_of_rows Matrix on the host with the data stored on the device.
* @param number_of_columns an integer.
* @param number_of_rows an integer.
*/
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

/**
* Creates a Matrix on the device which handles the same pointers as the intermediary Matrix.
*/
template <class elem_type>
void MatrixManager<elem_type>::initialise_device()
{
	HANDLE_ERROR( cudaMalloc( (void**) &device_ptr, sizeof( Matrix<elem_type> ) ) );

	//Copy the matrix data device pointers from the intermediary matrix to the device matrix.
	//Note now the device pointers are accessible from the intermediary matrix object and the device matrix object.
	HANDLE_ERROR( cudaMemcpy( device_ptr, intermediary_ptr, sizeof( Matrix<elem_type> ), cudaMemcpyHostToDevice ) );
}

/**
* Moves the pointers of the data stored in another MatrixManager object to this one.
*/
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

/**
* Moves the pointers of the data stored in Matrix object and creates the appropriate missing
* Matrices.
*/
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

/**
* Frees the memory used by the matrices manged by this MatrixManager object.
*/
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

/**
* Returns a reference to the intermediary Matrix object.
*/
template <class elem_type>
Matrix<elem_type>& MatrixManager<elem_type>::intermediary()
{
	return *intermediary_ptr;
}

/**
* Creates a square NxN matrix on the host and device.
* @param N a 4 byte integer.
*/
template <class elem_type>
MatrixManager<elem_type>::MatrixManager(const std::uint32_t N)
{
	initialise_host( N, N );
	initialise_intermediary( N, N );
	initialise_device();
}

/**
* Creates a rectangular number_of_columns x number_of_rows matrix on the host and device.
* @param number_of_columns a 4 byte integer.
* @param number_of_rows a 4 byte integer.
*/
template <class elem_type>
MatrixManager<elem_type>::MatrixManager( const uint32_t number_of_columns, const uint32_t number_of_rows )
{
	initialise_host( number_of_columns, number_of_rows );
	initialise_intermediary( number_of_columns, number_of_rows );
	initialise_device();
}

/**
* Frees the memory used by the Matrices managed by this MatrixManager object.
*/
template <class elem_type>
MatrixManager::~MatrixManager()
{
	clear();
}

/**
* Moves the pointers to the Matrices managed by the passed MatrixManager to the one being constructed.
* @param that a MatrixManager object.
*/
template <class elem_type>
MatrixManager<elem_type>::MatrixManager( MatrixManager<elem_type>&& that )
{
	move_data( that );
}

/**
* Moves the pointers to the Matrices managed by the MatrixManager being constructed.
* @param that a MatrixManager object.
*/
template<class elem_type>
MatrixManager<elem_type>::MatrixManager( Matrix<elem_type>&& that )
{
	move_data( that );
}

/**
* Moves the pointers to the Matrix on the RHS to the MatrixManager on the LHS
* @param that a MatrixManager object.
*/
template<class elem_type>
MatrixManager<elem_type>& MatrixManager<elem_type>::operator=( Matrix<elem_type>&& that )
{
	move_data( that );
	return *this;
}

/**
* Moves data from the Matrix object passed into ones intialised and managed by the MatrixManager object being constructed.
* @param that a Matrix object.
*/
template<class elem_type>
MatrixManager<elem_type>& MatrixManager<elem_type>::operator=( Matrix<elem_type>&& that )
{
	move_data( that );
	return *this;
}

/**
* Access to the Matrix data stored on the host.
* @return reference to a Matrix.
*/
template <class elem_type>
Matrix<elem_type>& MatrixManager<elem_type>::host() 
{ 
	return *host_ptr; 
}

/**
* Access to the Matrix data stored on the device.
* @return reference to a Matrix.
*/
template <class elem_type>
Matrix<elem_type>& Matrix<elem_type>::device() 
{ 
	return *device_ptr; 
}

/**
* Copies the values of the elements of the matrix to the device from the host.
*/
template <class elem_type>
void MatrixManager<elem_type>::host2device()
{
	HANDLE_ERROR( cudaMalloc( intermediary().data_ptr, host().get.data_ptr(), sizeof( elem_type ) * host().get.length(), cudaMemcpyHostToDevice ) );
}

/**
* Copies the values of the elements of the matrix to the host from the device.
*/
template <class elem_type>
void MatrixManager<elem_type>::device2host()
{
	HANDLE_ERROR( cudaMalloc( host().data_ptr, intermediary().get.data_ptr(), sizeof( elem_type ) * host().get.length(), cudaMemcpyDeviceToHost ) );
}