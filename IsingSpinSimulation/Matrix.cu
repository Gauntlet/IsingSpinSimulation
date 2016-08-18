#include "DataStructures.h"

using namespace kspace;

template<class elem_type> Matrix::initialize( const size_t numofcols, const size_t numofrows, const MemoryLocation memloc ) {

	const size_t tmplength = numofcols * numofrows;

	if ( memloc == MemoryLocation::host )
	{
		_memloc = new MemoryLocation();
		_data = new elem_type[ tmplength ]();
		_length = new size_t();
		_numOfCols = new size_t();
		_numOfRows = new size_t();

		( *_memloc ) = memloc;
		( *_length ) = tmplength;
		( *_numOfCols ) = numofcols;
		( *_numOfRows ) = numofrows;
	}
	else if ( memloc == MemroyLoc::device )
	{
		HANDLE_ERROR( cudaMalloc( (void**) &_memloc, sizeof( MemoryLocation ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_data, sizeof( elem_type )*tmplength ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_length, sizeof( size_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_numOfCols, sizeof( size_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_numOfRows, sizeof( size_t ) ) );

		HANDLE_ERROR( cudaMemcpy( _memlock, &memloc, sizeof( MemoryLocation ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemset( _data, 0, sizeof( elem_type )*tmplength ) );
		HANDLE_ERROR( cudaMemcpy( _length, &tmplength, sizeof( size_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _numOfCols, &numofcols, sizeof( size_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _numOfRows, &numofrows, sizeof( size_t ), cudaMemcpyHostToDevice ) );
	}
}

template<class elem_type> Matrix::Matrix( const size_t N, const MemoryLocation memloc )
{
	initialize( N, N, memloc );
}

template<class elem_type> Matrix::Matrix( const size_t M, const size_t N, const MemoryLocation memloc )
{
	initialize( M, N, memloc );
}

template<class elem_type> Matrix::~Matrix()
{
	if ( memLoc() == MemoryLocation::host )
	{
		delete _memloc;
		delete[] _data;
		delete _length;
		delete _numOfCols;
		delete _numOfRows;
	}
	else if ( memLoc() == MemroyLoc::device )
	{
		HANDLE_ERROR( cudaFree( _memloc ) );
		HANDLE_ERROR( cudaFree( _data ) );
		HANDLE_ERROR( cudaFree( _length ) );
		HANDLE_ERROR( cudaFree( _numOfCols ) );
		HANDLE_ERROR( cudaFree( _numOfRows ) );
	}
}

template<class elem_type>
CUDA_CALLABLE_MEMBER Matrix::MemoryLocation memLoc() const
{
	return *_memloc;
}

template<class elem_type>
CUDA_CALLABLE_MEMBER elem_type Matrix::get( const size_t row, const size_t col ) const
{
	assert( row >= 0 && row < numOfRows() && col >= 0 && col < numOfCols() );
	return _data[ row * numOfColumns() + col ];
}

template<class elem_type>
CUDA_CALLABLE_MEMBER void Matrix::set( const size_t row, const size_t col, const elem_type value )
{
	assert( row >= 0 && row < numOfRows() && col >= 0 && col < numOfCols() );
	_data[ row * numOfColumns() + col ] = value;
}

template<class elem_type>
CUDA_CALLABLE_MEMBER size_t Matrix::length() const
{
	return *_length;
}

template<class elem_type>
CUDA_CALLABLE_MEMBER size_t Matrix::numOfColumns() const
{
	return *_numOfCols;
}

template<class elem_type>
CUDA_CALLABLE_MEMBER size_t Matrix::numOfRows() const
{
	return *_numOfRows;
}