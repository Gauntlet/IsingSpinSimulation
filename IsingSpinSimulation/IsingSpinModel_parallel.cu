#include "Ising_Spin_Model.h"
#include "device_launch_parameters.h"

using namespace kspace;
using namespace IsingSpinModel;
using namespace detail;

__global__ void next_spin_change( const Graph::Graph* graph, Matrix<std::int8_t>* spin_states, curandState* globalState, const double beta, const size_t time_step )
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if ( n < graph->number_of_nodes() )
	{
		double flip_prob = calcFlipProb( graph, spin_states, beta, time_step, n );
		double RANDOM = generateIID( globalState, n );

		if ( RANDOM <= flip_prob )
		{
			spin_states->set( n, time_step + 1, -spin_states->get( n, time_step ) );
		}
	}
}

void IsingSpinModel_parallel::run( const std::uint32_t time_step, const double temperature )
{
	double beta = calculate_beta( temperature );

	next_spin_change( _graph.device, _spin_states.device, _global_states, beta, time_step );
}