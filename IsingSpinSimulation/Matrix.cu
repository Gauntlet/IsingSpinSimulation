#include "DataStructures.h"

using namespace kspace;

template<class elem_type> Matrix::initialize( const uint32_t numofcols, const uint32_t numofrows, const MemoryLocation memloc ) {

	const uint32_t tmplength = numofcols * numofrows;

	if ( memloc == MemoryLocation::host )
	{
		_memloc = new MemoryLocation();
		_data = new elem_type[ tmplength ]();
		_length = new uint32_t();
		_numOfCols = new uint32_t();
		_numOfRows = new uint32_t();

		( *_memloc ) = memloc;
		( *_length ) = tmplength;
		( *_numOfCols ) = numofcols;
		( *_numOfRows ) = numofrows;
	}
	else if ( memloc == MemroyLoc::device )
	{
		HANDLE_ERROR( cudaMalloc( (void**) &_memloc, sizeof( MemoryLocation ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_data, sizeof( elem_type )*tmplength ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_length, sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_numOfCols, sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_numOfRows, sizeof( uint32_t ) ) );

		HANDLE_ERROR( cudaMemcpy( _memlock, &memloc, sizeof( MemoryLocation ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemset( _data, 0, sizeof( elem_type )*tmplength ) );
		HANDLE_ERROR( cudaMemcpy( _length, &tmplength, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _numOfCols, &numofcols, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _numOfRows, &numofrows, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	}
}

template<class elem_type> Matrix::Matrix( const uint32_t N, const MemoryLocation memloc )
{
	initialize( N, N, memloc );
}

template<class elem_type> Matrix::Matrix( const uint32_t M, const uint32_t N, const MemoryLocation memloc )
{
	initialize( M, N, memloc );
}

template<class elem_type> Matrix::~Matrix()
{
	if ( memory_location() == MemoryLocation::host )
	{
		delete _memloc;
		delete[] _data;
		delete _length;
		delete _numOfCols;
		delete _numOfRows;
	}
	else if ( memory_location() == MemroyLoc::device )
	{
		HANDLE_ERROR( cudaFree( _memloc ) );
		HANDLE_ERROR( cudaFree( _data ) );
		HANDLE_ERROR( cudaFree( _length ) );
		HANDLE_ERROR( cudaFree( _numOfCols ) );
		HANDLE_ERROR( cudaFree( _numOfRows ) );
	}
}

template<class elem_type>
CUDA_CALLABLE_MEMBER Matrix::MemoryLocation memory_location() const
{
	return *_memloc;
}

template<class elem_type>
CUDA_CALLABLE_MEMBER elem_type Matrix::get( const uint32_t row, const uint32_t col ) const
{
	if (row < 0 || row >= numOfRows() || col <= 0 || col >= numOfCols())
	{
		throw std::invalid_argument("matrix indices out of bounds")
	}

	return _data[ row * numOfColumns() + col ];
}

template<class elem_type>
CUDA_CALLABLE_MEMBER void Matrix::set( const uint32_t row, const uint32_t col, const elem_type value )
{
	if (row < 0 || row >= numOfRows() || col <= 0 || col >= numOfCols())
	{
		throw std::invalid_argument("matrix indices out of bounds")
	}

	_data[ row * numOfColumns() + col ] = value;
}

template<class elem_type>
CUDA_CALLABLE_MEMBER uint32_t Matrix::length() const
{
	return *_length;
}

template<class elem_type>
CUDA_CALLABLE_MEMBER uint32_t Matrix::numOfColumns() const
{
	return *_numOfCols;
}

template<class elem_type>
CUDA_CALLABLE_MEMBER uint32_t Matrix::numOfRows() const
{
	return *_numOfRows;
}