#ifndef ISING_SPIN_MODEL_H
#define ISING_SPIN_MODEL_H

#include <random>
#include "Physical_Model.h"
#include "DataStructures.h"
#include "Graph.h"
#include "curand.h"
#include "curand_kernel.h"

namespace kspace
{
	class IsingSpinModelBase : public Physical_Model
	{
	protected:
		MatrixShared<std::int8_t> _spin_states;
		Graph::GraphShared _graph;
		curandState* _global_states;
	public:
		IsingSpinModelBase( Graph::GraphShared &graphshared, const std::uint32_t run_time, const std::uint32_t seed );
		~IsingSpinModelBase()
		{
			cudaFree( _global_states );
		}

		void reset_random_states( const uint32_t seed );
	};

	class IsingSpinModel : public IsingSpinModelBase
	{
	public:
		IsingSpinModel( Graph::GraphShared &graphshared, const std::uint32_t run_time, const std::uint32_t seed ) : IsingSpinModelBase( graphshared, run_time, seed ) {};
		void run( const std::uint32_t time_step, const double temperature );
	};

	class IsingSpinModel_MetropolisHastings : public Physical_Model
	{
	private:
		Graph::Graph _graph;
		Matrix<std::int8_t> _spin_states;

		std::uint32_t _seed;
		std::mt19937 _mt_node_selector;
		std::mt19937 _mt_flip_acceptor;

	protected:
		double calcAcceptanceProb( const std::int32_t node, const double beta, const std::uint32_t time_step ) const;
		void flip( const std::uint32_t time_step, const std::int32_t node );
	public:
		IsingSpinModel_MetropolisHastings( Graph::Graph &graph, const std::uint32_t run_time, const std::uint32_t seed1, const std::uint32_t seed2 ) : Physical_Model( run_time ), _graph( graph ), _spin_states( run_time, graph.number_of_nodes ), _seed( seed1 ), _mt_node_selector( seed2 ) {};
		void run( const std::uint32_t time_step, const std::uint32_t temperature );
	};

	namespace detail
	{
		#define BOLTZMANN_CONSTANT 1.38064852e-23

		double calculate_beta( const double temperature )
		{
			return ( ( double ) 1. / ( BOLTZMANN_CONSTANT * temperature ) );
		}

		__device__ double generateIID( curandState* globalState, int n )
		{
			//Generate a random number between 0 and 1.
			curandState localState = globalState[ n ];
			double RANDOM = curand_uniform_double( &localState );
			globalState[ n ] = localState;

			return RANDOM;
		}

		__device__ size_t countNeighbourMatches( const Graph::Graph* graph, const Matrix<std::int8_t>* spin_states, const size_t time_step, const int n )
		{
			int s = 0;
			int m = -1;
			for ( int k = 0; k < graph->degree( n ); ++k )
			{
				m = graph->neighbour( n, k );
				if ( spin_states->get( n, time_step ) == spin_states->get( m, time_step ) )
				{
					s++;
				}
			}

			return s;
		}

		__device__ double calcFlipProb( const Graph::Graph* graph, const Matrix<std::int8_t>* spin_states, const double beta, const size_t time_step, const int n )
		{
			//Calculate the probability of vertex n flipping state.
			int S = countNeighbourMatches( graph, spin_states, time_step, n );

			double numerator = std::exp( -2 * beta * ( 2 * S - graph->degree( n ) ) );
			double denominator = 1. + numerator;
		}
	}
}

#endif