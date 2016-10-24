#include "JaggedList.h"
#include <numeric>

using namespace kspace;

/**
* Moves the pointers managed by the JaggedList passed to the JaggedList which called this function.
* @param that a JaggedList.
*/
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

/**
* Creates N lists each with a length specified in 'lengths' on the device or host as specified.
* @param N an integer.
* @param lengths a pointer to an array of length N.
* @param memloc an enum.
*/
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

/**
* Frees the memory managed by the JaggedList container.
*/
template <class elem_type>
JaggedList<elem_type>::~JaggedList()
{
	set.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
* Returns whether the data is stored on the device or host memory.
* @return an enum.
*/
template <class elem_type>
MemoryLocation JaggedList<elem_type>::JAGGED_LIST_GET::memory_location() const
{
	return parent.memloc;
}

/**
* Read only access to the array storing the data.
* @return a pointer to a const elem_type.
*/
template <class elem_type>
elem_type const * JaggedList<elem_type>::JAGGED_LIST_GET::data_ptr() const
{
	return parent.data_ptr;
}

/**
* Read only access to the array storing the list lengths.
* @return a pointer to a const uint32_t.
*/
template <class elem_type>
std::uint32_t const * JaggedList<elem_type>::JAGGED_LIST_GET::lengths_ptr() const
{
	return parent.lengths_ptr;
}

/**
* Read only access to the array storing the offsets.
* @return a pointer to a const uint32_t.
*/
template <class elem_type>
uint32_t const * JaggedList<elem_type>::JAGGED_LIST_GET::offsets_ptr() const
{
	return parent.offsets_ptr;
}

/**
* Read only access to an element in a specified list.
*
* Both the list_id and index are checked to see if they are out of range.
* @param list_id an integer.
* @param index an integer.
* @return a reference to a const elem_type.
*/
template <class elem_type>
elem_type const & JaggedList<elem_type>::JAGGED_LIST_GET::operator()( uint32_t const list_id, uint32_t const index ) const
{
	if ( list_id >= size() || index >= length( list_id ) )
	{
		throw std::out_of_range( "Jagged List indices are out of range." );
	}

	return parent.data_ptr[ offset( list_id ) + index ];
}

/**
* Read only access to the number of lists.
* @return an integer.
*/
template <class elem_type>
uint32_t const & JaggedList<elem_type>::JAGGED_LIST_GET::size() const
{
	return *parent.size;
}

/**
* Read only access to the number of elements in a specified list.
* @param list_id an integer.
* @return an integer.
*/
template <class elem_type>
uint32_t const & JaggedList<elem_type>::JAGGED_LIST_GET::length( uint32_t const list_id ) const
{
	if ( list_id >= size() )
	{
		throw std::out_of_range( "Jagged List list_id is out of range." );
	}

	return *parent.size_flat;
}

/**
* Read only access to the offset at which a specified list starts in the array of elements.
* @param list_id an integer.
* @return an integer.
*/
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

/**
* Read and write access to the array storing the data.
* @return a pointer to a const elem_type.
*/
template <class elem_type>
elem_type* JaggedList<elem_type>::JAGGED_LIST_SET::data_ptr()
{
	return parent.data_ptr;
}

/**
* Read and write access to the array storing the list lengths.
* @return a pointer to a const uint32_t.
*/
template <class elem_type>
uint32_t* JaggedList<elem_type>::JAGGED_LIST_SET::lengths_ptr()
{
	return parent.lengths_ptr;
}

/**
* Read and write access to the array storing the offsets.
* @return a pointer to a const uint32_t.
*/
template <class elem_type>
uint32_t* JaggedList<elem_type>::JAGGED_LIST_SET::offsets_ptr()
{
	return parent.offsets_ptr;
}

/**
* Read and write access to an element in a specified list.
*
* Both the list_id and index are checked to see if they are out of range.
* @param list_id an integer.
* @param index an integer.
* @return a reference to a const elem_type.
*/
template <class elem_type>
elem_type& JaggedList<elem_type>::JAGGED_LIST_SET::operator()( const uint32_t list_id, const uint32_t index )
{
	if ( list_id >= parent.get.size() || index >= parent.get.length( list_id ) )
	{
		throw std::out_of_range( "Jagged List indices are out of range." );
	}

	return parent.data_ptr[ parent.get.offset( list_id ) + index ];
}

/**
* Frees the resources managed by the JaggedList.
*/
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