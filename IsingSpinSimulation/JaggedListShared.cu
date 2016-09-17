#include "DataStructures.h"

using namespace kspace;

template <class elem_type>
JaggedListShared<elem_type>::JaggedListShared( const uint32_t N, const uint32_t* lengths )
{
	host = new JaggedList( N, lengths, MemoryLocation::host );
	intermediary = new JaggetList( N, lengths, MemoryLocation::device );
	HANDLE_ERROR( cudaMalloc( (void**) &device, sizeof( JaggedList ) ) );
	HANDLE_ERROR( cudaMemcpy( device, intermediary, sizeof( JaggedList ), cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMemcpy( intermediary->_memloc, host->_memloc, sizeof( MemoryLocation ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( intermediary->_data, host->_data, sizeof( elem_type )*host->size(), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( intermediary->_length, host->_length, sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( intermediary->_lengths, host->_lengths, sizeof( uint32_t )*host->length(), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( intermediary->_offsets, host->_offsets, sizeof( uint32_t )*( host->length() + 1 ), cudaMemcpyHostToDevice ) );
}

template <class elem_type>
JaggedListShared::JaggedListShared()
{
	HANDLE_ERROR( cudaFree( device ) );
	delete host;
	delete intermediary;
}

template <class elem_type>
void kspace::JaggedListShared<elem_type>::host2device()
{
	HANDLE_ERROR( cudaMemcpy( intermediary->_data, host->_data, sizeof( elem_type )*host->size(), cudaMemcpyHostToDevice ) );
}

template <class elem_type>
void JaggedListShared<elem_type>::device2host()
{
	HANDLE_ERROR( cudaMemcpy( host->_data, intermediary->_data, sizeof( elem_type )*host->size(), cudaMemcpyDeviceToHost ) );
}