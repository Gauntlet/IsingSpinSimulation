#include "JaggedList.h"
#include <numeric>

using namespace kspace;

template <class elem_type>
void JaggedList<elem_type>::move_data( JaggedList<elem_type>&& that )
{
	set.clear();

	data_ptr = std::move( that.data_ptr );
	lengths_ptr = std::move( that.lengths_ptr );
	offsets_ptr = std::move( that.offsets_ptr );
	size = std::move( that.size );
	size_flat = std::move( that.size_flat );
	memloc = std::move( that.memloc );

	that.data_ptr = nullptr;
	that.lengths_ptr = nullptr;
	that.offsets_ptr = nullptr;
	that.size = nullptr;
	that.size_flat = nullptr;
	that.memloc = MemoryLocation::host;
}

template <class elem_type>
JaggedList<elem_type>::JaggedList( uint32_t const &  N, uint32_t const * lengths, MemoryLocation const & memloc ) : get( *this ), set( *this )
{
	this->memloc = memloc;

	uint32_t* tmpoffsets = new uint32_t[ N + 1 ]();
	std::partial_sum( lengths, lengths + N, tmpoffsets + 1 );

	if ( memloc == MemoryLocation::host )
	{
		data_ptr = new elem_type[ tmpoffsets[ N ] ]();

		this->lengths = new uint32_t[ N ]();

		offsets_ptr = std::move( tmpoffsets );
		tmpoffsets = nullptr;

		size = new uint32_t();
		size_flat = new uint32_t();

		( *length ) = offsets_ptr[ N ];
		( *size ) = N;
		memcpy( (void*) lengths, (void*) lengths, sizeof( uint32_t ) * N );
	}
	else if ( memloc == MemoryLocation::device )
	{
		HANDLE_ERROR( cudaMalloc( (void**) &data, sizeof( elem_type ) * ( tmpoffsets[ N ] ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &length, sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( (void**) &lengths, sizeof( uint32_t ) * N ) );
		HANDLE_ERROR( cudaMalloc( (void**) &offsets, sizeof( uint32_t ) * ( N + 1 ) ) );

		HANDLE_ERROR( cudaMemset( data, 0, sizeof( elem_type ) * tmpoffsets[ N ] ) );

		HANDLE_ERROR( cudaMemcpy( length, &N, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( (void*) lengths, (void*) lengths, sizeof( uint32_t ) * N, cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( offsets, tmpoffsets, sizeof( uint32_t ) * ( N + 1 ), cudaMemcpyHostToDevice ) );
	}

	if ( nullptr == tmpoffsets )
	{
		delete[] tmpoffsets;
	}
}

template <class elem_type>
JaggedList<elem_type>::~JaggedList()
{
	( *this ).set.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class elem_type>
MemoryLocation JaggedList<elem_type>::JAGGED_LIST_GET::memory_location() const
{
	return parent.memloc;
}

template <class elem_type>
elem_type const * JaggedList<elem_type>::JAGGED_LIST_GET::data_ptr() const
{
	return parent.data_ptr;
}

template <class elem_type>
std::uint32_t const * JaggedList<elem_type>::JAGGED_LIST_GET::lengths_ptr() const
{
	return parent.lengths_ptr;
}

template <class elem_type>
uint32_t const * JaggedList<elem_type>::JAGGED_LIST_GET::offsets_ptr() const
{
	return parent.offsets_ptr;
}

template <class elem_type>
elem_type const & JaggedList<elem_type>::JAGGED_LIST_GET::operator()( uint32_t const list_id, uint32_t const index ) const
{
	if ( list_id >= size() || index >= length( list_id ) )
	{
		throw std::out_of_range( "Jagged List indices are out of range." );
	}

	return parent.data_ptr[ offset( list_id ) + index ];
}

template <class elem_type>
uint32_t const & JaggedList<elem_type>::JAGGED_LIST_GET::size() const
{
	return *parent.size;
}

template <class elem_type>
uint32_t const & JaggedList<elem_type>::JAGGED_LIST_GET::length( uint32_t const list_id ) const
{
	if ( list_id >= size() )
	{
		throw std::out_of_range( "Jagged List list_id is out of range." );
	}

	return *parent.size_flat;
}

template <class elem_type>
uint32_t const & JaggedList<elem_type>::JAGGED_LIST_GET::offset( const uint32_t list_id ) const
{
	if ( list_id >= size() )
	{
		throw std::out_of_range( "Jagged List list_id is out of range." );
	}

	return parent.offsets_ptr[ list_id ];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class elem_type>
elem_type* JaggedList<elem_type>::JAGGED_LIST_SET::data_ptr()
{
	return parent.data_ptr;
}

template <class elem_type>
uint32_t* JaggedList<elem_type>::JAGGED_LIST_SET::lengths_ptr()
{
	return parent.lengths_ptr;
}

template <class elem_type>
uint32_t* JaggedList<elem_type>::JAGGED_LIST_SET::offsets_ptr()
{
	return parent.offsets_ptr;
}

template <class elem_type>
elem_type& JaggedList<elem_type>::JAGGED_LIST_SET::operator()( const uint32_t list_id, const uint32_t index )
{
	if ( list_id >= parent.get.size() || index >= parent.get.length( list_id ) )
	{
		throw std::out_of_range( "Jagged List indices are out of range." );
	}

	return parent.data_ptr[ parent.get.offset( list_id ) + index ];
}

template <class elem_type>
void JaggedList<elem_type>::JAGGED_LIST_SET::clear()
{
	if ( memory_location() == MemoryLocation::host )
	{
		delete[] parent.data_ptr;
		delete[] parent.lengths_ptr;
		delete[] parent.offsets_ptr;
		delete parent.size;
		delete parent.size_flat;
	}
	else if ( memory_location() == MemoryLocation::device )
	{
		HANDLE_ERROR( cudaFree( data ) );
		HANDLE_ERROR( cudaFree( length ) );
		HANDLE_ERROR( cudaFree( size ) );
		HANDLE_ERROR( cudaFree( lengths ) );
		HANDLE_ERROR( cudaFree( offsets ) );
	}
}