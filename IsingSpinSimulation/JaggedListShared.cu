#include "DataStructures.h"

using namespace kspace;

template <class elem_type> 
JaggedListShared::JaggedListShared( const size_t N, const size_t* lengths )
{
	host = new JaggedList( N, lengths, MemoryLocation::host );
	intermediary = new JaggetList( N, lengths, MemoryLocation::device );
	cudaMalloc( (void**) &device, sizeof( JaggedList ) );
	cudaMemcpy( device, intermediary, sizeof( JaggedList ), cudaMemcpyHostToDevice );
	
	cudaMemcpy( intermediary->_memloc, host->_memloc, sizeof( MemoryLocation ), cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary->_data, host->_data, sizeof( elem_type )*host->size(), cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary->_length, host->_length, sizeof( size_t ), cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary->_lengths, host->_lengths, sizeof( size_t )*host->length(), cudaMemcpyHostToDevice );
	cudaMemcpy( intermediary->_offsets, host->_offsets, sizeof( size_t )*( host->length() + 1 ), cudaMemcpyHostToDevice );
}

template <class elem_type>
JaggedListShared::JaggedListShared()
{
	cudaFree(device);
	delete host;
	delete intermediary;
}

template <class elem_type>
void JaggedListShared::host2device()
{
	cudaMemcpy( intermediary->_data,	host->_data,	sizeof( elem_type )*host->size(),			cudaMemcpyHostToDevice );
}

template <class elem_type>
void JaggedListShared::device2host()
{
	cudaMemcpy( host->_data,	intermediary->_data,	sizeof( elem_type )*host->size(),			cudaMemcpyDeviceToHost );
}