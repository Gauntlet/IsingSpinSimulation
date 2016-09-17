#include "DataStructures.h"
#include <numeric>

using namespace kspace;

template <class elem_type>
JaggedList<elem_type>::JaggedList( const uint32_t N, const uint32_t* lengths, const MemoryLocation memloc )
{
	_memloc = memloc;

	uint32_t* tmpoffsets = new uint32_t[ N + 1 ]();
	std::partial_sum( lengths, lengths + N, tmpoffsets + 1 );

	if ( memloc == MemoryLocation::host )
	{
		_data = new elem_type[ tmpoffsets[ N ] ]();
		_length = new uint32_t();
		_size = new uint32_t();
		_lengths = new uint32_t[ N ]();
		_offsets = new uint32_t[ N + 1 ]();

		( *_length ) = tmpoffsets[ N ];
		( *_size ) = N;
		memcpy( _lengths, lengths, sizeof( uint32_t ) * N );
		memcpy( _offsets, tmpoffsets, sizeof( uint32_t ) * ( N + 1 ) );
	}
	else if ( memloc == MemoryLocation::device )
	{
		HANDLE_ERROR( cudaMalloc( (void**) &_data, sizeof( elem_type ) * ( tmpoffsets[ N ] ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_length, sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_lengths, sizeof( uint32_t ) * N ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_offsets, sizeof( uint32_t ) * ( N + 1 ) ) );

		HANDLE_ERROR( cudaMemset( _data, 0, sizeof( elem_type ) * tmpoffsets[ N ] ) );
		HANDLE_ERROR( cudaMemcpy( _length, &N, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _lengths, lengths, sizeof( uint32_t ) * N, cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _offsets, tmpoffsets, sizeof( uint32_t ) * ( N + 1 ), cudaMemcpyHostToDevice ) );
	}

	delete[] tmpoffsets;
}

template <class elem_type>
JaggedList<elem_type>::~JaggedList()
{
	if ( memory_location() == MemoryLocation::host )
	{
		delete[] _data;
		delete _length;
		delete _size;
		delete[] _lengths;
		delete[] _offsets;
	}
	else if ( memory_location() == MemoryLocation::device )
	{
		HANDLE_ERROR( cudaFree( _data ) );
		HANDLE_ERROR( cudaFree( _length ) );
		HANDLE_ERROR( cudaFree( _size ) );
		HANDLE_ERROR( cudaFree( _lengths ) );
		HANDLE_ERROR( cudaFree( _offsets ) );
	}
}

template <class elem_type>
MemoryLocation JaggedList<elem_type>::memory_location() const
{
	return *_memloc;
}

template <class elem_type>
elem_type* JaggedList<elem_type>::raw_data()
{
	return *_data;
}

template <class elem_type>
uint32_t* JaggedList<elem_type>::raw_degrees()
{
	return *degrees;
}

template <class elem_type>
uint32_t* JaggedList<elem_type>::raw_offsets()
{
	return *_offsets;
}

template <class elem_type>
elem_type JaggedList<elem_type>::get( const uint32_t row, const uint32_t col ) const
{
	if ( row < 0 || row >= numOfRows() || col <= 0 || col >= numOfCols() )
	{
		throw std::invalid_argument( "jagged list indices out of bounds" );
	}

	return _data[ offset( row ) + col ];
}

template <class elem_type>
void JaggedList<elem_type>::set( const uint32_t row, const uint32_t col, const elem_type val )
{
	if ( row < 0 || row >= numOfRows() || col <= 0 || col >= numOfCols() )
	{
		throw std::invalid_argument( "jagged list indices out of bounds" );
	}

	_data[ offset( row ) + col ] = val;
}

template <class elem_type>
uint32_t JaggedList<elem_type>::length() const
{
	return *_length;
}

template <class elem_type>
uint32_t JaggedList<elem_type>::size() const
{
	return *_size;
}

template <class elem_type>
uint32_t JaggedList<elem_type>::length( const uint32_t row ) const
{
	if ( row < 0 || row >= numOfRows() )
	{
		throw std::invalid_argument( "jagged list rows index out of bounds" );
	}

	return _lengths[ row ];
}

template <class elem_type>
uint32_t JaggedList<elem_type>::offset( const uint32_t row ) const
{
	if ( row < 0 || row >= numOfRows() )
	{
		throw std::invalid_argument( "jagged list rows index out of bounds" );
	}

	return _offsets[ row ];
}