#include "Matrix.h"

using namespace kspace;

/**
* Allocates space in memory (host or device) for the matrix data.
* It is a helper function to simplify the constructors.
* @param number_of_columns an integer number.
* @param number_of_rows an integer number.
* @param memloc an enum that indicates the memory location to store the data.
*/
template<class elem_type>
void Matrix<elem_type>::initialise( const uint32_t num_of_columns, const uint32_t num_of_rows, const MemoryLocation memloc )
{
	_memloc = memloc;

	const uint32_t tmplength = num_of_columns * num_of_rows;
	if ( memloc == MemoryLocation::host )
	{
		_data = new elem_type[ tmplength ]();
		_length = new uint32_t();
		_number_of_columns = new uint32_t();
		_number_of_rows = new uint32_t();

		( *_length ) = tmplength;
		( *_number_of_columns ) = num_of_columns;
		( *_number_of_rows ) = num_of_rows;
	}
	else if ( memloc == MemroyLoc::device )
	{
		HANDLE_ERROR( cudaMalloc( (void**) &_data, sizeof( elem_type ) * tmplength ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_length, sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_number_of_columns, sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_number_of_rows, sizeof( uint32_t ) ) );

		HANDLE_ERROR( cudaMemset( _data, 0, sizeof( elem_type ) * tmplength ) );
		HANDLE_ERROR( cudaMemcpy( _length, &tmplength, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _number_of_columns, &num_of_columns, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _number_of_rows, &num_of_rows, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	}
}

/**
* A helper function that moves the pointers stored in the Matrix object passed as a parameter
* into this Matrix object.
* @param that is a Matrix object with the same element type as the object it is being passed into.
*/
template <class elem_type>
void Matrix<elem_type>::move_data( Matrix<elem_type>&& that )
{
	delete[] data;
	delete length;
	delete number_of_columns;
	delete number_of_rows;

	memlock = that.memloc;
	data = that.data;
	length = that.length;
	number_of_columns = that.number_of_columns;
	number_of_rows = that.number_of_rows;

	that.memloc = NULL;
	that.data = nullptr;
	that.length = nullptr;
	that.number_of_columns = nullptr;
	that.number_of_rows = nullptr;
}

/**
* Creates a Matrix that stores data of elem_type but is completely empty.
* Data can be moved into this container at a later time.
*/
template<class elem_type>
void Matrix<elem_type>::Matrix() : get(*this), set(*this), memloc(MemoryLocation::host), data_ptr(nullptr), length(nullptr), number_of_columns(nullptr), number_of_rows(nullptr) {}

/**
* Creates a NxN square matrix on the host or device as indicated.
* @param N an integer number that defines the size of the matrices rows and columns.
* @param memloc an enum that indicates whether the data is stored on host or device memory.
*/
template<class elem_type>
Matrix<elem_type>::Matrix( const uint32_t N, const MemoryLocation memloc ) : get( *this ), set( *this )
{
	initialize( N, N, memloc );
}

/**
* Creates a number_of_columns x number_of_rows matrix on the host or device as indicated.
* @param number_of_columns an integer number that defines the size of the columns.
* @param number_of_rows an integer number that defines the size of the rows.
* @param memloc an enum that indicates whether the data is stored on host or device memory.
*/
template<class elem_type>
Matrix<elem_type>::Matrix( const uint32_t num_of_columns, const uint32_t num_of_rows, const MemoryLocation memloc ) : get( *this ), set( *this )
{
	initialize( num_of_columns, num_of_rows, memloc );
}

/**
* A destructor.
* Will take free the data stored by the object, whether it is on the device or host memory.
*/
template<class elem_type>
Matrix<elem_type>::~Matrix()
{
	set.clear();
}

/**
* Define a move constructor.
* Move the pointers stored in the passed Matrix<elem_type> into the one being constructed.
* @param that a Matrix object with the same element type as the one being constructed.
*/
template <class elem_type>
Matrix::Matrix( Matrix<elem_type>&& that ) : get( *this ), set( *this )
{
	move_data( that );
}

/**
* Define a move assignement operator.
* Move the pointers stored in the RHS to the one on the LHS.
* @param that a Matrix object on the RHS.
* @return a Matrix object with data of the same element type.
*/
template <class elem_type>
Matrix<elem_type>& Matrix::operator=( Matrix<elem_type>&& that )
{
	move_data( that );
	return *this;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* Indicates whether the data is stored on device or host memory.
* @return an enum.
*/
template <class elem_type>
MemoryLocation const & Matrix<elem_type>::MATRIX_GET::memory_location() const
{
	return parent.memloc;
}

/**
* Read only access to a single element of the matrix.
* @param column an integer.
* @param row an integer.
* @return reference to const element.
*/
template <class elem_type>
elem_type const & Matrix<elem_type>::MATRIX_GET::operator()( const size_t column, const size_t row ) const
{
	if ( column >= number_of_columns() || row >= number_of_rows() )
	{
		throw std::out_of_range( "Matrix indices out of range." );
	}

	return parent.data_ptr[ column * number_of_rows() + row ];
}

/**
* Read only access to all the elements in the matrix.
* A pointer to the first element in the matrix. The matrix is stored as a 1D array using column order.
* @return pointer to const elements.
*/
template <class elem_type>
elem_type const * Matrix<elem_type>::MATRIX_GET::data_ptr() const
{
	return parent.data_ptr;
}

/**
* Read only access to a column of the matrix.
* @param column an integer.
* @return pointer to a const column of elements.
*/
template <class elem_type>
elem_type const * Matrix<elem_type>::MATRIX_GET::data_ptr( const std::uint32_t column ) const
{
	if ( column >= number_of_columns() )
	{
		throw std::out_of_range( "Column index is greater than number of columns" );
	}

	return parent.data_ptr + column*number_of_rows();
}

/**
* Read only access to the number of columns in the matrix.
* @return an 4 byte integer.
*/
template <class elem_type>
uint32_t const & Matrix<elem_type>::MATRIX_GET::number_of_columns() const
{
	return parent.number_of_columns;
}

/**
* Read only access to the number of rows in the matrix.
* @return an 4 byte integer.
*/
template <class elem_type>
uint32_t const & Matrix<elem_type>::MATRIX_GET::number_of_rows() const
{
	return parent.number_of_rows;
}

/**
* Read only access to the total number of elements in the matrix.
* @return an 4 byte integer.
*/
template <class elem_type>
uint32_t const & Matrix<elem_type>::MATRIX_GET::length() const
{
	return parent.length;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* Read and Write access to a single element of the matrix.
* @param column an integer.
* @param row an integer.
* @return reference to element.
*/
template <class elem_type>
elem_type& Matrix<elem_type>::MATRIX_SET::operator()( const size_t column, const size_t row ) const
{
	if ( column >= parent.get.number_of_columns() || row >= parent.get.number_of_rows() )
	{
		throw std::out_of_range( "Matrix indices out of range." );
	}

	return parent.data_ptr[ column*parent.get.number_of_rows() + row ];
}

/**
* Read and write access to all the elements in the matrix.
* A pointer to the first element in the matrix. The matrix is stored as a 1D array using column order.
* @return pointer to elements.
*/
template <class elem_type>
elem_type* Matrix<elem_type>::MATRIX_SET::data_ptr() const
{
	return parent.data_ptr;
}

/**
* Read and write access to a column of the matrix.
* @param column an integer.
* @return pointer to a column of elements.
*/
template <class elem_type>
elem_type* Matrix<elem_type>::MATRIX_SET::data_ptr( const std::uint32_t column ) const
{
	return parent.data_ptr + column*parent.get.number_of_rows();
}

/**
* Frees the memory being managed by the Matrix object.
*/
template <class elem_type>
void Matrix<elem_type>::MATRIX_SET::clear() const
{
	if ( MemoryLocation::host == memloc )
	{
		delete[] data_ptr;
		delete length;
		delete number_of_columns;
		delete number_of_rows;
	}
	else if ( MemoryLocation::device == memloc )
	{
		HANDLE_ERROR( cudaFree( data ) );
		HANDLE_ERROR( cudaFree( length ) );
		HANDLE_ERROR( cudaFree( number_of_columns ) );
		HANDLE_ERROR( cudaFree( number_of_rows ) );
	}
}