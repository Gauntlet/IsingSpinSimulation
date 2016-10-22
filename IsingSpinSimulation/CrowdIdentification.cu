#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <thrust\scan.h>
#include <thrust\device_ptr.h>


#include <algorithm>
#include "CrowdIdentification.h"
#include "details.h"
#include "Array.h"

using namespace kspace;

/**
* A group of functions needed to process the matrix of spin states into spin cluster ids.
*
* Importantly the functions have been designed such that they can be called by both device and host processes. 
* This means that they can be run in parallel given that the appropriate data structures are passed in as parameters.
*/
class HELPER
{
public:
	/**
	* A simple template function for checkign whether the values of two variables are the same.
	*/
	template <class T>
	CUDA_CALLABLE_MEMBER static size_t kronecker_delta( T const & a, T const & b )
	{
		return (size_t) ( a == b );
	}

	/**
	* Returns the smoothened spin state of a node given its current spin state, the average number of times it flipped in a period of time
	* and its average state during that same period.
	*/
	CUDA_CALLABLE_MEMBER static std::int8_t smooth_state( std::int8_t const & spin_state, double const & flip_average, double const & average_state )
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
			else if ( average_state >( 1. / 3. ) )
			{
				return 1;
			}
		}

		return spin_state;
	}

	/**
	* Calculates the flip and state average of a given node for each time step.
	*/
	CUDA_CALLABLE_MEMBER static void smoother_helper( std::int32_t const & node, std::uint32_t const & N, std::uint32_t const & T, std::uint32_t const & window_size, std::int8_t const * spin_states, std::int8_t * smoothed_states )
	{
		size_t flip_count = 0;
		size_t average_state = 0;
		double F, A;
		std::int8_t s1, s2;


		for ( size_t t = 0; t < window_size; ++t )
		{
			s1 = *( spin_states + N*t + node );
			s2 = *( spin_states + N*t + N + node );

			flip_count += 1 - HELPER::kronecker_delta( s1, s2 );
			average_state += s1;
		}

		for ( int t = 0; t < T; ++t )
		{
			size_t lb = std::max( t - window_size, (size_t) 0 );
			size_t ub = std::min( t + window_size, T );

			//Subtract the contribution that has left the averaging window.
			if ( lb > 0 )
			{
				s1 = *( spin_states + N*lb - N + node );
				s2 = *( spin_states + N*lb + node );

				flip_count -= 1 - HELPER::kronecker_delta( s1, s2 );
				average_state -= s1;
			}

			//Add the contribution that has entered the averaging window.
			if ( ub < T )
			{
				s1 = *( spin_states + N*ub - N + node );
				s2 = *( spin_states + N*ub + node );

				flip_count += 1 - HELPER::kronecker_delta( s1, s2 );
				average_state += s2;
			}

			F = (double) flip_count / ( ub - lb );
			A = (double) average_state / ( 2 * window_size + 1 );

			*( smoothed_states + N*t + node ) = HELPER::smooth_state( *( spin_states + N*t + node ), F, A );
		}
	}

	/**
	* This is BFS that does not use std::queue or vector. Thus it can be used on both the host and device.
	*/
	CUDA_CALLABLE_MEMBER static void partitioner( GRAPH::Graph const & graph, std::int8_t const * spin_states, bool* visited, std::uint32_t* queue, std::uint32_t* offsets, std::uint32_t* partitions, std::int32_t & number_of_partitions )
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

	/**
	* Computes which partition in the previous time step each node is linked to. Nodes which are in the same partition at different times are
	* part of the same spin cluster.
	*/
	CUDA_CALLABLE_MEMBER static void linker( std::uint32_t const & number_of_nodes, std::int32_t const & number_of_partitions, std::int8_t const * spin_states, std::uint32_t const * partitions, std::uint32_t const * offsets, std::uint32_t* similarities, std::int32_t* linklist, std::int32_t & unlinked_count )
	{
		uint32_t p1, p2;

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
			if ( ( spin_states - number_of_nodes )[ n ] == spin_states[ n ] )
			{
				p1 = ( partitions - number_of_nodes )[ n ];
				p2 = partitions[ n ];

				//Iterate over the linklist checking to see if a link between partition ids
				//at t and t+1 have already been linked and creating a link if they have not.
				//The similarity between the two partitions is also incremented.
				for ( int offset = offsets[ p2 ]; offset < offsets[ p2 + 1 ]; ++offset )
				{
					//If the partition of the node stored in the linklist is the same as the 
					//partition p.first increment the similarity between them by 1.
					if ( ( partitions - number_of_nodes )[ linklist[ offset ] ] == p1 )
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

		std::uint32_t p1size, p2size;
		std::int32_t n;
		bool islinked;

		//Iterate over each partition.
		for ( int P = 0; P < number_of_partitions; ++P )
		{
			//Initialise islinked to false.
			islinked = false;

			//Set the partition being considered at time step t.
			p2 = P;
			//Compute the size of the partition p.first.
			p2size = offsets[ p2 + 1 ] - offsets[ p2 ];

			//Iterate over all of the potential partitions at time step t+1 to link p.first to.
			for ( int offset = offsets[ p2 ]; offset < offsets[ p2 + 1 ]; ++offset )
			{
				//Set the partition being considered at time step t.
				n = linklist[ offset ];
				p1 = ( partitions - number_of_nodes )[ n ];
				//Compute the size the partition p.first.
				p1size = ( offsets - number_of_nodes )[ p1 + 1 ] - ( offsets - number_of_nodes )[ p1 ];

				//Check that the similarity is greater than 0.5.
				//If there is a similarity greater than 0.5 it is unique and thus 
				//any other similarities can be disregarded
				if ( (double) similarities[ offset ] / (double) ( p1size + p2size - similarities[ offset ] ) > 0.5 )
				{
					//We store the node id of a node in the previous time step as when we relabel the partition_ids
					//into spin_cluster_ids we will not have to update the list with new spin_cluster_ids as they 
					//change.
					linklist[ p2 ] = n;

					islinked = true;

					//If a similarity greater than 0.5 is found then we exit the loop as it is unique.
					break;
				}

			}

			//If a link is not found for this partition then set its link as null (== number_of_nodes).
			//At the same time count the number of unlinked partitions.
			if ( !islinked )
			{
				linklist[ offsets[ p2 ] ] = number_of_nodes;
				unlinked_count++;
			}
		}
	}

	/**
	* In some cases partitions at time step t can not be linked to any in the previous time step. In that case they are a
	* a new spin cluster and need to be assigned a new spin_cluster_id. This function finds such partitions and does that.
	*/
	CUDA_CALLABLE_MEMBER static void unlinked_linker( std::uint32_t const & number_of_nodes, std::uint32_t const & number_of_partitions, std::int32_t const & unlinked_count, std::int32_t* linklist )
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

	/**
	* Relabels the partition ids of each node into the appropriate spin cluster id.
	*/
	CUDA_CALLABLE_MEMBER static void relabeller( std::int32_t const & node, std::uint32_t const & number_of_nodes, std::uint32_t const & wT, std::int32_t const * linklists, std::uint32_t * partitions )
	{
		std::int32_t L;
		std::uint32_t offset;

		//We iterate over the t time steps relabelling the partition_ids
		//to spin_cluster_ids.
		for ( int t = 0; t < wT; ++t )
		{
			offset = number_of_nodes * t;
			L = linklists[ offset + partitions[ offset + node ] ];
			if ( L < 0 )
			{
				//If the link id 'L' is less than 0 this means that it has
				//no link in the previous time step. Which means it is a 
				//new spin cluster, thus relabel the current partition_id with 
				//the absolute of L as the spin_cluster_id.
				partitions[ offset + node ] = std::abs( L );
			}
			else
			{
				//If the link id 'L' is greater than or equal to 0 then it
				//is the node id of a spin_cluster_id in the previous time 
				//step that P is linked to. Thus we relabel the current node's
				//partition_id to that spin_cluster_id.
				partitions[ offset + node ] = partitions[ offset - number_of_nodes + L ];
			}
		}
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* Performs a single threaded smoothening of the spin states.
* @param spin_states reference to a Matrix<int8_t>.
* @param window_size a size_t integer. The flip and state averages are computed in the interval t-windows_size <= t <= t+window_size.
*/
Matrix<std::int8_t>&& CrowdIdentification::CROWD_IDENTIFICATION_HOST::smoothen( Matrix<std::int8_t>& spin_states, size_t const window_size )
{
	const size_t N = spin_states.get.number_of_rows();
	const size_t T = spin_states.get.number_of_columns();

	Matrix<std::int8_t> smoothed_states( T, N, MemoryLocation::host );

	for ( size_t n = 0; n < N; ++n )
	{
		HELPER::smoother_helper( n, N, T, window_size, spin_states.get.data_ptr(), smoothed_states.set.data_ptr() );
	}

	return std::move(smoothed_states);
}

/**
* Performs a single threaded partitioning of the graph according the spin states.
* 
* Each partition is a connected subgraph where every node has the same spin state. The partitions are maximal (as large as they can be while
* meeting the rules which define the partitioning).
*/
Matrix<std::uint32_t>&& CrowdIdentification::CROWD_IDENTIFICATION_HOST::partition( GRAPH::Graph const & graph, Matrix<std::int8_t>& smooth_spin_states )
{
	std::uint32_t N = graph.get.number_of_nodes();
	std::uint32_t T = smooth_spin_states.get.number_of_columns();

	Matrix<std::uint32_t> partitions( T, N, MemoryLocation::host );
	Matrix<std::int32_t> linklists( T, N, MemoryLocation::host );

	{
		//During each time step partition the graph into spin clusters.
		Array<std::uint32_t> offsets( N + 1 );
		Array<bool> visited( N );
		Array<std::uint32_t> queue( N );
		Array<std::uint32_t> similarities( N );

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
		HELPER::partitioner( graph, smooth_spin_states.get.data_ptr( 0 ), visited.set.data_ptr(), queue.set.data_ptr(), offsets.set.data_ptr(), partitions.set.data_ptr( 0 ), number_of_partitions );

		//Set the number of unlinked during the first time step to
		//the number of partitions. Then each successive unlinked
		//partition will be automatically given a unique spin_cluster_ids
		//by reading this number and incrementing it.
		unlinked_count = number_of_partitions;

		for ( size_t t = 1; t < smooth_spin_states.get.number_of_columns(); ++t )
		{
			//Compute the partitions of the t-th time step using BFS algorithm.
			HELPER::partitioner( graph, smooth_spin_states.get.data_ptr( t ), visited.set.data_ptr(), queue.set.data_ptr(), offsets.set.data_ptr(), partitions.set.data_ptr( t ), number_of_partitions );

			//Compute the links between partitions which will be used to form the spin clusters.
			//Also get the number of unlinked partitions in this time step.
			unlinked_count_tmp = 0;
			HELPER::linker( graph.get.number_of_nodes, number_of_partitions, smooth_spin_states.get.data_ptr( t ), partitions.get.data_ptr( t ), offsets.get.data_ptr(), similarities.set.data_ptr(), linklists.set.data_ptr( t ), unlinked_count_tmp );

			//Iterate over the partitions and if there are any unlinked partitions.
			//Give them a unique spin_cluster_id using unlinked_count.
			if ( unlinked_count_tmp > 0 )
			{
				HELPER::unlinked_linker( graph.get.number_of_nodes(), number_of_partitions, unlinked_count, linklists.set.data_ptr( t ) );
				//Increment unlinked_count by the number of unlinked partitions in this time step.
				unlinked_count += unlinked_count_tmp;
			}
		}
	}

	for ( std::uint32_t node = 0; node < graph.get.number_of_nodes(); ++node )
	{
		//For each node iterate over each time step and relabel its partition_id to a spin_cluster_id.
		HELPER::relabeller( node, N, T, linklists.get.data_ptr(), partitions.set.data_ptr() );
	}

	return std::move(partitions);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* A CUDA kernel for smoothening the spin states.
* @param N a 32bit unsigned integer. The number of nodes in the graph.
* @param T a 32bit unsigned integer. The number of time steps spin states were computed for.
* @param spin_states a pointer to the 1D array containing the spin states.
* @param window_size a 32bit unsigned integer. The number of time steps on either side of t used for computing the smoothed spin state.
* @param smoothed_states an pointer to the 1D array to which the smoothed states are written.
*/
__global__ void smoothen_device( std::uint32_t const &N, std::uint32_t const &T, std::int8_t const * spin_states, std::uint32_t const & window_size, std::int8_t * smoothed_states )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < N )
	{
		HELPER::smoother_helper( n, N, T, window_size, spin_states, smoothed_states );
	}
}

/**
* Manages the resources required to smoothen the spin states on the device.
*/
MatrixManager<std::int8_t>&& CrowdIdentification::CROWD_IDENTIFICATION_DEVICE::smoothen( Matrix<std::int8_t>& spin_states, std::uint32_t const & window_size )
{
	MatrixManager<std::int8_t> smoothed_states( spin_states.get.number_of_columns(), spin_states.get.number_of_rows() );

	smoothen_device( spin_states.get.number_of_rows(), spin_states.get.number_of_columns(), spin_states.get.data_ptr(), window_size, smoothed_states.device().set.data_ptr() );

	return std::move(smoothed_states);
}

/**
* A CUDA kernel for computing the paritions for a period of time 'wT'. This is required as VRAM tends to be smaller than RAM, thus is not capable of storing
* very large matrices. By computing the partitions in blocks we can continue to work with large graphs and long simulation times.
* @param N a 32bit unsigned integer. The number of nodes in the graph.
* @param wT a 32bit unsigned integer. The size of the period of time for which the partitions are being computed.
* @param graph a pointer to the graph stored on the device. Necessary for partition the spin states into spin clusters.
* @param spin_states a pointer to a matrix represented as 1D array of the spin states.
* @param visited a pointer to a matrix of bools represented as 1D array. By default the values are false and set to true when a node has been visited during the BFS algorithm.
* @param queue a pointer to a matrix of unsigned integers represented as 1D array. Node ids are added to the queue one after another when the BFS algorithm visits them.
* @return offsets a pointer to a matrix of unsigned integers represented as a 1D array. These indicate the position in the similarity matrix a particular partitions similarities begin.
* @return partitions a pointer to a matrix of signed integers represented as a 1D array. The element (t,n) is the partition id of the node n at time t.
* @return number_of_partitions a pointer to a 1D array of unsigned integers. Keeps track of the number of partitions in each time step.
*/
__global__ void partitioner_device( std::uint32_t const & N, std::uint32_t const & wT, GRAPH::Graph const * graph, std::int8_t const * spin_states, bool* visited, std::uint32_t* queue, std::uint32_t* offsets, std::uint32_t* partitions, std::int32_t * number_of_partitions )
{
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if ( t < wT )
	{
		HELPER::partitioner( *graph, spin_states + N*t, visited + N*t, queue + N*t, offsets + N*t, partitions + N*t, *( number_of_partitions + t ) );
	}
}

/**
* A CUDA kernel that computes which partitions in consecutive time steps are linked, i.e. part of the same spin cluster.
* @param N a 32bit unsigned integer. The number of nodes in the graph.
* @param wT a 32bit unsigned integer. The size of the period of time for which the partitions are being computed.
* @param number_of_partitions a pointer to a 1D array of unsigned integers. The number of partitions in each time step.
* @param spin_states a pointer to a matrix of 8bit integers represented as 1D array of the spin states.
* @param partitions a pointer to a matrix of signed integers represented as a 1D array. The element (t,n) is the partition id of the node n at time t.
* @param offsets a pointer to a matrix of unsigned integers represented as a 1D array. These indicate the position in the similarity matrix a particular partitions similarities begin.
* @param similarities a pointer to a matrix of 32bit unsigned integers represented as 1D array. The elements are the number of nodes that are the same in a pair of partitions in consecutive timesteps.
* @return linklist a pointer to a matrix of 32bit signed integers represented as 1D array. The element (t, p) is the node id of a node in a partition P that the nodes in partition p at time t are linked to.
* @return unlinked_count a pointer to a 1D array of 32bit signed integers. Each element is the number of partitions for which there is no link to a partition in the previous time step.
*/
__global__ void linker_device( std::uint32_t const & N, std::uint32_t const & wT, std::int32_t const * number_of_partitions, std::int8_t const * spin_states, std::uint32_t const * partitions, std::uint32_t const * offsets, std::uint32_t* similarities, std::int32_t* linklist, std::int32_t * unlinked_count )
{
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if ( t < wT )
	{
		HELPER::linker( N, *( number_of_partitions + t ), spin_states + N*t, partitions + N*t, offsets + t, similarities + t, linklist + t, *( unlinked_count + t ) );
	}
}

/**
* Assigns unique spin_cluster_ids for each partition which does not have a link.
* @param N a 32bit unsigned integer. The number of nodes in the graph.
* @param wT a 32bit unsigned integer. The size of the period of time for which the partitions are being computed.
* @param number_of_partitions a pointer to a 1D array of unsigned integers. The number of partitions in each time step.
* @param unlinked_count a pointer to a 1D array of 32bit signed integers. Each element is the cumulative sum of the unlinked partitions up to the previous time step.
* @return linklist a pointer to a matrix of 32bit signed integers represented as 1D array. The element (t, p) is the node id of a node in a partition P that the nodes in partition p at time t are linked to.
*/
__global__ void unlinked_linker_device( std::uint32_t const & N, std::uint32_t const & wT, std::int32_t const * number_of_partitions, std::int32_t const * unlinked_count, std::int32_t* linklist )
{
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if ( t < wT )
	{
		HELPER::unlinked_linker( N, *( number_of_partitions + t ), *( unlinked_count + t ), linklist + N*t );
	}
}

/**
* Relabels the partition_id of each node to it spin_cluster_id.
* @param N a 32bit unsigned integer. The number of nodes in the graph.
* @param wT a 32bit unsigned integer. The size of the period of time for which the partitions are being computed.
* @return linklist a pointer to a matrix of 32bit signed integers represented as 1D array. The element (t, p) is the node id of a node in a partition P that the nodes in partition p at time t are linked to.
* @param partitions a pointer to a matrix of signed integers represented as a 1D array. The element (t,n) is the partition id of the node n at time t.
*/
__global__ void relabeller_device( std::uint32_t const & N, std::uint32_t const & wT, std::int32_t const * linklists, std::uint32_t* partitions )
{
	int n = blockDim.x + blockIdx.x + threadIdx.x;

	if ( n < N )
	{
		HELPER::relabeller( n, N, wT, linklists, partitions );
	}
}

/**
* Manages the resources for processing the smoothened_spin_states and graph structure to obtain a matrix containing the spin clusters.
* @param graph a reference to a const GraphManager. This contains a graph which is accessible from both host and device processes and is read only.
* @param smooth_spin_States a reference to a const MatrixManager<std::int8_t>. This contains a matrix the elements of which are availble to both device and host processes and is read only.
* @param time_block_size a 32bit unsigned integer. The size of the block that should be processed at one time. Used to limit the memory useage on the device.
*/
MatrixManager<std::uint32_t>&& CrowdIdentification::CROWD_IDENTIFICATION_DEVICE::partition( GRAPH::GraphManager const & graph, MatrixManager<std::int8_t> & smooth_spin_states, std::uint32_t const & time_block_size )
{
	const std::uint32_t N = smooth_spin_states.host().get.number_of_rows();
	const std::uint32_t T = smooth_spin_states.host().get.number_of_columns();

	MatrixManager<std::uint32_t> partitions( T, N );

	//We create the following matrices with time_block_size columns, to limit the amount memory
	//being allocated on the device.
	//We can then launch the kernels to work on a block of the partitions at a time.
	//This works as the relabeller relies on the previous time step already being
	//labelled with the correct spin_cluster_id or a new spin_cluster_id being assigned
	//already.
	Matrix<bool> visited( time_block_size, N, MemoryLocation::device );
	Matrix<std::uint32_t> queue( time_block_size, N, MemoryLocation::device );
	Matrix<std::int32_t> linklists( time_block_size, N, MemoryLocation::device );
	Matrix<std::uint32_t> offsets( time_block_size, N, MemoryLocation::device );
	Matrix<std::uint32_t> similarities( time_block_size, N, MemoryLocation::device );

	Matrix<std::int32_t> number_of_partitions( T, 1, MemoryLocation::device );
	MatrixManager<std::int32_t> unlinked_count( T, 1 );

	//Compute the partitions of the 0-th time step using BFS algorithm.
	partitioner_device( N, 1, graph.get.device_ptr(), smooth_spin_states.device().get.data_ptr(), visited.set.data_ptr(), queue.set.data_ptr(), offsets.set.data_ptr(), partitions.device().set.data_ptr(), number_of_partitions.set.data_ptr() );
	//Set the number of unlinked partitions in the 0-th time step as the number of nodes (N)
	//as they do not link back to anything.
	cudaMemcpy( unlinked_count.device().set.data_ptr(), &N, sizeof( std::int32_t ), cudaMemcpyHostToDevice );
	thrust::device_ptr<std::int32_t> d_ptr;

	for ( std::uint32_t lT = 1; lT < T; lT += time_block_size )
	{
		size_t offset = N*lT;

		partitioner_device( N, time_block_size, graph.get.device_ptr(), smooth_spin_states.device().get.data_ptr() + offset, visited.set.data_ptr(), queue.set.data_ptr(), offsets.set.data_ptr(), partitions.device().set.data_ptr() + offset, number_of_partitions.set.data_ptr() + lT );

		linker_device( N, time_block_size, number_of_partitions.get.data_ptr() + lT, smooth_spin_states.device().get.data_ptr() + offset, partitions.device().get.data_ptr() + offset, offsets.get.data_ptr(), similarities.set.data_ptr(), linklists.set.data_ptr(), unlinked_count.device().set.data_ptr() + lT );

		//Use the thrust library to do an in place prefix-sum (cumulative sum).
		//We first cast the raw device pointer to a device_ptr so that the function knows
		//that the data is on the device.
		d_ptr = thrust::device_pointer_cast<std::int32_t>( unlinked_count.device().set.data_ptr() );
		thrust::exclusive_scan( d_ptr + lT - 1, d_ptr + lT + time_block_size, d_ptr + lT - 1 );

		unlinked_linker_device( N, time_block_size, number_of_partitions.get.data_ptr() + lT, unlinked_count.device().get.data_ptr() + lT, linklists.set.data_ptr() );

		relabeller_device( N, time_block_size, linklists.get.data_ptr(), partitions.device().set.data_ptr() + offset );
	}

	return std::move(partitions);
}