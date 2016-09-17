#include "DataStructures.h"

using namespace kspace;

template<class elem_type>
void Matrix<elem_type>::initialise( const uint32_t num_of_columns, const uint32_t num_of_rows, const MemoryLocation memloc ) {
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

template<class elem_type> Matrix<elem_type>::Matrix( const uint32_t N, const MemoryLocation memloc )
{
	initialize( N, N, memloc );
}

template<class elem_type> Matrix<elem_type>::Matrix( const uint32_t num_of_columns, const uint32_t num_of_rows, const MemoryLocation memloc )
{
	initialize( num_of_columns, num_of_rows, memloc );
}

template<class elem_type> Matrix<elem_type>::~Matrix()
{
	if ( memory_location() == MemoryLocation::host )
	{
		delete[] _data;
		delete _length;
		delete _number_of_columns;
		delete _number_of_rows;
	}
	else if ( memory_location() == MemroyLoc::device )
	{
		HANDLE_ERROR( cudaFree( _data ) );
		HANDLE_ERROR( cudaFree( _length ) );
		HANDLE_ERROR( cudaFree( _number_of_columns ) );
		HANDLE_ERROR( cudaFree( _number_of_rows ) );
	}
}

template<class elem_type>
Matrix<elem_type>::MemoryLocation memory_location() const
{
	return _memloc;
}

template<class elem_type>
elem_type* Matrix<elem_type>::raw_data()
{
	return _data;
}

template<class elem_type>
elem_type* Matrix<elem_type>::raw_data( const std::uint32_t column )
{
	return _data + number_of_rows() * column;
}

template<class elem_type>
elem_type Matrix<elem_type>::get( const uint32_t column, const uint32_t row ) const
{
	if ( row < 0 || row >= number_of_rows() || column <= 0 || column >= number_of_columns() )
	{
		throw std::invalid_argument( "matrix indices out of bounds" )
	}

	return _data[ column * number_of_rows + row ];
}

template<class elem_type>
void Matrix<elem_type>::set( const uint32_t column, const uint32_t row, const elem_type value )
{
	if ( row < 0 || row >= number_of_rows() || column <= 0 || column >= number_of_columns() )
	{
		throw std::invalid_argument( "matrix indices out of bounds" )
	}

	_data[ column * number_of_rows + row ] = value;
}

template<class elem_type>
uint32_t Matrix<elem_type>::length() const
{
	return *_length;
}

template<class elem_type>
uint32_t Matrix<elem_type>::number_of_columns() const
{
	return *_number_of_columns;
}

template<class elem_type>
uint32_t Matrix<elem_type>::number_of_rows() const
{
	return *_number_of_rows;
}