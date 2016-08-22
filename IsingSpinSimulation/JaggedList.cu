#include "DataStructures.h"
#include <numeric>

using namespace kspace;

template <class elem_type>
JaggedList<elem_type>::JaggedList( const uint32_t N, const uint32_t* lengths, const MemoryLocation memloc )
{
	uint32_t* tmpoffsets = new uint32_t[ N + 1 ]();
	std::partial_sum( lengths, lengths + N, tmpoffsets + 1 );

	if ( memloc == MemorLocation::host )
	{
		_memloc = new MemoryLocation();
		_data = new elem_type[ tmpoffsets[ N ] ]();
		_length = new uint32_t();
		_lengths = new uint32_t[ N ]();
		_offsets = new uint32_t[ N + 1 ]();

		( *_memloc ) = memloc;
		( *_length ) = N;
		memcpy( _lengths, lengths, sizeof( uint32_t )*N );
		memcpy( _offsets, tmpoffsets, sizeof( uint32_t )*( N + 1 ) );
	}
	else if ( memloc == MemoryLocation::device )
	{
		HANDLE_ERROR( cudaMalloc( (void**) &_memloc, sizeof( MemoryLocation ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_data, sizeof( elem_type )*( tmpoffsets[ N ] ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_length, sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_lengths, sizeof( uint32_t )*N ) );
		HANDLE_ERROR( cudaMalloc( (void**) &_offsets, sizeof( uint32_t )*( N + 1 ) ) );

		HANDLE_ERROR( cudaMemcpy( _memloc, &memloc, sizeof( MemoryLocation ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemset( _data, 0, sizeof( elem_type ) * tmpoffsets[N] ) );
		HANDLE_ERROR( cudaMemcpy( _length, &N, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _lengths, lengths, sizeof( uint32_t ) * N, cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( _offsets, tmpoffsets, sizeof( uint32_t ) * (N+1), cudaMemcpyHostToDevice ) );
	}

	delete[] tmpoffsets;
}

template <class elem_type>
JaggedList<elem_type>::~JaggedList()
{
	if ( memLoc() == MemorLocation::host )
	{
		delete _memloc;
		delete[] _data;
		delete _length;
		delete[] _lengths;
		delete[] _offsets;
	}
	else if ( memLoc() == MemoryLocation::device )
	{
		HANDLE_ERROR( cudaFree( _memloc ) );
		HANDLE_ERROR( cudaFree( _data ) );
		HANDLE_ERROR( cudaFree( _length ) );
		HANDLE_ERROR( cudaFree( _lengths ) );
		HANDLE_ERROR( cudaFree( _offsets ) );
	}
}

template <class elem_type>
CUDA_CALLABLE_MEMBER MemoryLocation JaggedList<elem_type>::memLoc() const
{
	return *_memloc;
}

template <class elem_type>
CUDA_CALLABLE_MEMBER elem_type JaggedList<elem_type>::get( const uint32_t row, const uint32_t col ) const
{
	assert( row >= 0 && row < length() && col >= 0 && col < lengths(row) );
	return _data[ offset( row ) + col ];
}

template <class elem_type>
CUDA_CALLABLE_MEMBER void JaggedList<elem_type>::set( const uint32_t row, const uint32_t col, const elem_type val)
{
	assert( row >= 0 && row < length() && col >= 0 && col < lengths( row ) );
	_data[ offset( row ) + col ] = val;
}

template <class elem_type>
CUDA_CALLABLE_MEMBER uint32_t JaggedList<elem_type>::length() const
{
	return *_length;
}

template <class elem_type>
CUDA_CALLABLE_MEMBER uint32_t JaggedList<elem_type>::size() const
{
	return _offsets[ length() ];
}

template <class elem_type>
CUDA_CALLABLE_MEMBER uint32_t JaggedList<elem_type>::length( const uint32_t row ) const
{
	assert( row >= 0 && row < length() );
	return *_lengths[ row ];
}

template <class elem_type>
CUDA_CALLABLE_MEMBER uint32_t JaggedList<elem_type>::offset( const uint32_t row ) const
{
	assert( row >= 0 && row < length() );
	return *_offsets[row];
}