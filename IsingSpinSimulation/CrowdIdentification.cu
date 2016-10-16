#include "CrowdIdentification.h"
#include <algorithm>
#include "DataStructures.h"
#include "ArrayHandle.h"

using namespace kspace;

template <class T>
size_t CrowdIdentification::kronecker_delta( T const & a, T const & b )
{
	return (size_t) ( a == b );
}

std::int8_t CrowdIdentification::smooth_state( std::int8_t const & spin_state, double const & flip_average, double const & average_state )
{
	if ( flip_average >= 0.5 )
	{
		return 0;
	}
	else if ( flip_average < 0.5 )
	{
		if ( average_state < ( -1. / 3. ) )
		{
			return -1;
		}
		else if ( average_state > ( 1. / 3. ) )
		{
			return 1;
		}
	}

	return spin_state;
}

//This is BFS that does not use std::queue or vector. Thus it can be used on both the host and device.
void CrowdIdentification::partitioner( GRAPH::Graph const & graph, std::int8_t const * spin_states, bool* visited, std::uint32_t* queue, std::uint32_t* offsets, std::uint32_t* partitions, std::int32_t & number_of_partitions )
{
	//Initialise the visited and queue arrays to zero.
	std::memset( visited, 0, graph.get.number_of_nodes() );
	std::memset( queue, 0, graph.get.number_of_nodes() );

	//Initialise the queue getter and setter.
	size_t queue_front( 0 ), queue_back( 0 );

	//Declare the current node (v) and the neighbour (w).
	std::int32_t v, w;

	//Initialise the partition_id at 0.
	std::uint32_t partition_id = 0;

	//Initialise the offset at 0.
	std::uint32_t offset = 0;

	//Iterate through each node.
	for ( int i = 0; i < graph.get.number_of_nodes(); ++i )
	{
		//If a node has not been visited yet start a BFS from that node.
		if ( !visited[ i ] )
		{
			//Label the node with the partition id.
			partitions[ i ] = partition_id;
			//Mark the node as visited.
			visited[ i ] = true;
			//Add the node to queue.
			queue[ queue_back ] = i;
			//Increment the queue setter.
			queue_back++;

			//Iteratre through the queue until the queue getter catches up with setter.
			while ( queue_front != queue_back )
			{
				//Get a node from the queue.
				v = queue[ queue_front ];
				//Increment the offset.
				offset++;

				//Iterate over node v's neighbours.
				for ( size_t k = 0; k < graph.get.degree( v ); ++k )
				{
					w = graph.get.neighbour( v, k );

					//Check if the neighbour (w) has not been visited.
					if ( !visited[ w ] )
					{
						//Check if the neighbour (w) has the same spin state as node v.
						if ( spin_states[ v ] == spin_states[ w ] )
						{
							//Set w's partition_id.
							partitions[ w ] = partition_id;
							//Set w as visited.
							visited[ w ] = true;

							//Add w to the queue.
							queue[ queue_back ] = w;
							//Increment the queue setter.
							queue_back++;
						}
					}
				}

				//Increment the queue getter.
				queue_front++;
			}

			//Increment partition_id.
			partition_id++;

			//Set the offset for the partition_id.
			if ( partition_id <= graph.get.number_of_nodes() )
			{
				offsets[ partition_id ] = offset;
			}
		}
	}

	number_of_partitions = partition_id;
}

void CrowdIdentification::linker( std::uint32_t const & number_of_nodes, std::int32_t const & number_of_partitions, std::pair<std::int8_t const *, std::int8_t const *> const & spin_states, std::pair<std::uint32_t const *, std::uint32_t const *> const &  partitions, std::uint32_t const * offsets, std::uint32_t* similarities, std::int32_t* linklist, std::int32_t & unlinked_count )
{
	std::pair<uint32_t, uint32_t> p;

	//The linker first calculates the similarities between partitions
	//in time step t-1 and t.
	//It ensures that similarities are calculated for valid pairs.
	//Consequently that means there is an upper bound of "number_of_nodes" possible
	//similarities to calculate.
	//Once all simialrities are computed we then run over all the partition pairs
	//and find the maximum similarities.
	//If the maximum similarity is also greater than 0.5 then this indicates there
	//is a unique link
	//Iterate over each node.

	for ( int n = 0; n < number_of_nodes; ++n )
	{
		//Check whether the spin state at time t is the same at t+1.
		if ( spin_states.first[ n ] == spin_states.second[ n ] )
		{
			p.first = partitions.first[ n ];
			p.second = partitions.second[ n ];

			//Iterate over the linklist checking to see if a link between partition ids
			//at t and t+1 have already been linked and creating a link if they have not.
			//The similarity between the two partitions is also incremented.
			for ( int offset = offsets[ p.second ]; offset < offsets[ p.second + 1 ]; ++offset )
			{
				//If the partition of the node stored in the linklist is the same as the 
				//partition p.first increment the similarity between them by 1.
				if ( partitions.first[ linklist[ offset ] ] == p.first )
				{
					similarities[ offset ]++;
					break;
				}
				else if ( number_of_nodes == linklist[ offset ] )
				{
					//If the node id stored in the linklist is null (where null == number_of_nodes)
					//then set the linklist to the current node id and set the similarity to 1.
					linklist[ offset ] = n;
					similarities[ offset ] = 1;
					break;
				}
			}
		}
	}

	std::pair<std::uint32_t, std::uint32_t> psize;
	double maxsim( 0 ), cursim( 0 );
	std::int32_t n;
	bool islinked;

	//Iterate over each partition.
	for ( int P = 0; P < number_of_partitions; ++P )
	{
		//Initialise the maxsim to zero.
		maxsim = 0;
		//Initialise islinked to false.
		islinked = false;

		//Set the partition being considered at time step t.
		p.second = P;
		//Compute the size of the partition p.first.
		psize.second = offsets[ p.second + 1 ] - offsets[ p.second ];

		//Iterate over all of the potential partitions at time step t+1 to link p.first to.
		for ( int offset = offsets[ p.second ]; offset < offsets[ p.second + 1 ]; ++offset )
		{
			//Set the partition being considered at time step t.
			n = linklist[ offset ];
			p.first = partitions.first[ n ];
			//Compute the size the partition p.first.
			psize.first = offsets[ p.first + 1 ] - offsets[ p.first ];
			//Compute the similarity of the two partitions.
			cursim = (double) similarities[ offset ] / (double) ( psize.first + psize.second - similarities[ offset ] );

			//If the similarity of the two partitions being considers is greater 
			//than the maximum pairing then set the considered pairing as the maximum one.
			//AND check that the similarity is greater than 0.5.
			if ( cursim > maxsim && cursim > 0.5 )
			{
				//We store the node id of a node in p.first as when we relabel the partitions
				//into their spin clusters we will not have to update the list with new partition ids.
				linklist[ p.second ] = n;
				maxsim = cursim;

				islinked = true;
				
				//If cursim > 0.5 it is unique there is no other partition pair which can have
				//greater similarity thus we end this loop.
				break;
			}

		}

		//If a link is not found for this partition then set its link as null (== number_of_nodes).
		//At the same time count the number of unlinked partitions.
		if ( !islinked )
		{
			int offset = offsets[ p.second ];
			linklist[ offset ] = number_of_nodes;
			unlinked_count++;
		}
	}
}

void CrowdIdentification::unlinked_linker( std::uint32_t const & number_of_nodes, std::uint32_t const & number_of_partitions, std::int32_t const & unlinked_count, std::int32_t* linklist )
{
	std::int32_t unlinked_count_tmp = unlinked_count;
	for ( int P = 0; P < number_of_partitions; ++P )
	{
		if ( number_of_nodes == linklist[ P ] )
		{
			//If a partition is not already linked then set a label using a negative value.
			//This will tell the relabeller to use the absolute value of this number as 
			// this partitions spin_cluster_id.
			linklist[ P ] = -unlinked_count_tmp;
			unlinked_count_tmp++;
		}
	}
}

void CrowdIdentification::relabeller( std::int32_t const & node, std::uint32_t const & number_of_nodes, std::uint32_t const & num_of_timesteps, std::int32_t const * linklists, std::uint32_t * partitions )
{
	std::uint32_t P;
	std::int32_t L;
	std::uint32_t offset;
	
	//We iterate over the t time steps relabelling the partition_ids
	//to spin_cluster_ids.
	for ( int t = 1; t < num_of_timesteps; ++t )
	{
		offset = number_of_nodes * t;
		P = partitions[offset + node];
		L = linklists[ P ];
		if ( L < 0 )
		{
			//If the link id 'L' is less than 0 this means that it has
			//no link in the previous time step. Which means it is a 
			//new spin cluster, thus relabel the current partition_id with 
			//the absolute of L as the spin_cluster_id.
			partitions[ offset + node] = std::abs( L );
		}
		else
		{
			//If the link id 'L' is greater than or equal to 0 then it
			//is the node id of a spin_cluster_id in the previous time 
			//step that P is linked to. Thus we relabel the current node's
			//partition_id to that spin_cluster_id.
			partitions[ offset + node ] = partitions[offset - number_of_nodes + L ];
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Matrix<std::int8_t> CrowdIdentification::HOST::smoothen( Matrix<std::int8_t>& spin_states, size_t const window_size )
{
	const size_t N = spin_states.get.number_of_rows();
	const size_t T = spin_states.get.number_of_columns();

	double F, A;

	Matrix<std::int8_t> smoothed_states( T, N, MemoryLocation::host );

	for ( size_t n = 0; n < N; ++n )
	{
		size_t flip_count = 0;
		size_t average_state = 0;
		for ( size_t t = 0; t < window_size; ++t )
		{
			flip_count += 1 - kronecker_delta( spin_states.get( t, n ), spin_states.get( t + 1, n ) );
			average_state += spin_states.get( t, n );
		}

		average_state += spin_states.get( window_size, n );

		for ( int t = 0; t < T; ++t )
		{
			size_t lb = std::max( t - window_size, (size_t) 0 );
			size_t ub = std::min( t + window_size, T );

			//Subtract the contribution that has left the averaging window.
			if ( lb > 0 )
			{
				flip_count -= 1 - kronecker_delta( spin_states.get( lb - 1, n ), spin_states.get( lb, n ) );
				average_state -= spin_states.get( lb - 1, n );
			}

			//Add the contribution that has entered the averaging window.
			if ( ub < T )
			{
				flip_count += 1 - kronecker_delta( spin_states.get( ub - 1, n ), spin_states.get( ub, n ) );
				average_state += spin_states.get( ub, n );
			}

			F = (double) flip_count / ( ub - lb );
			A = (double) average_state / ( 2 * window_size + 1 );

			smoothed_states.set( n, t ) = smooth_state( spin_states.get( n, t ), F, A );
		}
	}

	return smoothed_states;
}

void CrowdIdentification::HOST::partition( GRAPH::Graph const & graph, Matrix<std::int8_t>& smooth_spin_states )
{
	Matrix<std::uint32_t> partitions( smooth_spin_states.get.number_of_columns(), smooth_spin_states.get.number_of_rows(), MemoryLocation::host );
	Matrix<std::int32_t> linklists( smooth_spin_states.get.number_of_columns(), smooth_spin_states.get.number_of_rows(), MemoryLocation::host );

	{
		//During each time step partition the graph into spin clusters.
		ArrayHandle<std::uint32_t> offsets( smooth_spin_states.get.number_of_rows() + 1 );
		ArrayHandle<bool> visited( graph.get.number_of_nodes() );
		ArrayHandle<std::uint32_t> queue( graph.get.number_of_nodes() );
		ArrayHandle<std::uint32_t> similarities( smooth_spin_states.get.number_of_rows() );

		std::pair<std::int8_t const *, std::int8_t const *> spin_state;
		std::pair<std::uint32_t const *, std::uint32_t const *> partition;

		std::int32_t number_of_partitions;
		std::int32_t unlinked_count, unlinked_count_tmp;

		//During each time step t we compute the partitions in that time step.
		//Then we calculate the similarity of the partitions between t-1 and t.
		//We then create a linklist which points partitions at time step t to a
		//node in the linked partition at time step t-1.
		//This is done so that when we re-label partition ids using spin-cluster ids
		//the linklist will point to a node in the previous time step that has already
		//been relabelled with the correct spin-cluster id.
		//This allows us to "easily" parallelise the relabelling algorithm.

		//Compute the partitions of the 0-th time step using BFS algorithm.
		partitioner( graph, smooth_spin_states.get.data_ptr( 0 ), visited.set.data(), queue.set.data(), offsets.set.data, partitions.set.data_ptr( 0 ), number_of_partitions);

		//Set the number of unlinked during the first time step to
		//the number of partitions. Then each successive unlinked
		//partition will be automatically given a unique spin_cluster_ids
		//by reading this number and incrementing it.
		unlinked_count = number_of_partitions;

		for ( size_t t = 1; t < smooth_spin_states.get.number_of_columns(); ++t )
		{
			//Compute the partitions of the t-th time step using BFS algorithm.
			partitioner( graph, smooth_spin_states.get.data_ptr( t ), visited.set.data(), queue.set.data(), offsets.set.data, partitions.set.data_ptr( t ), number_of_partitions);

			//Get the spin states for the previous and current time steps.
			spin_state.first = smooth_spin_states.get.data_ptr( t - 1 );
			spin_state.second = smooth_spin_states.get.data_ptr( t );

			//Get the partitions for the previous and current time steps.
			partition.first = partitions.get.data_ptr( t - 1 );
			partition.second = partitions.get.data_ptr( t );

			//Compute the links between partitions which will be used to form the spin clusters.
			//Also get the number of unlinked partitions in this time step.
			unlinked_count_tmp = 0;
			linker( graph.get.number_of_nodes, number_of_partitions, spin_state, partition, offsets.get.data(), similarities.set.data(), linklists.set.data_ptr( t ), unlinked_count_tmp );

			//Iterate over the partitions and if there are any unlinked partitions.
			//Give them a unique spin_cluster_id using unlinked_count.
			if ( unlinked_count_tmp > 0 )
			{
				unlinked_linker( graph.get.number_of_nodes(), number_of_partitions, unlinked_count, linklists.set.data_ptr( t ) );
				//Increment unlinked_count by the number of unlinked partitions in this time step.
				unlinked_count += unlinked_count_tmp;
			}
		}
	}

	for ( std::uint32_t node = 0; node < graph.get.number_of_nodes(); ++node )
	{
		//For each node iterate over each time step and relabel its partition_id to a spin_cluster_id.
		relabeller( node, graph.get.number_of_nodes(), partitions.get.number_of_columns(), linklists.get.data_ptr(), partitions.set.data_ptr() );
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MatrixShared<std::int8_t> CrowdIdentification::DEVICE::smoothen( Matrix<std::int8_t>& spin_states, size_t const window_size )
{

}

void CrowdIdentification::DEVICE::partition( GRAPH::Graph const & graph, MatrixShared<std::uint32_t>& smooth_spin_states )
{

}