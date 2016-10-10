#include "DataStructures.h"
#include "Matrix.h"

using namespace kspace;

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

template<class elem_type>
Matrix<elem_type>::Matrix( const uint32_t N, const MemoryLocation memloc ) : get( *this ), set( *this )
{
	initialize( N, N, memloc );
}

template<class elem_type>
Matrix<elem_type>::Matrix( const uint32_t num_of_columns, const uint32_t num_of_rows, const MemoryLocation memloc ) : get( *this ), set( *this )
{
	initialize( num_of_columns, num_of_rows, memloc );
}

template<class elem_type>
Matrix<elem_type>::~Matrix()
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