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
	namespace IsingSpinModel
	{

		/*///////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////Spin State FILE STRUCTURE//////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////

		|---------------------------------------------------------------------------------------|
		|	OFFSET	|		HEADER FIELD		|	SIZE (B)	|	TYPE		|	VALUE		|
		|---------------------------------------------------------------------------------------|
		|	0		|	FILE ID					|		11		|	char		|	xSPINSTATEx	|
		|	11		|	Major Version			|		1		|	uint8_t		|	1			|
		|	12		|	Minor Version			|		1		|	uint8_t		|	0			|
		|	13		|	zlib Major Version		|		1		|	uint8_t		|	1			|
		|	14		|	zlib Minor Version		|		1		|	uint8_t		|	0			|
		|	15		|	zlib Build Version		|		1		|	uint8_t		|	0			|
		|	16		|	Parameter Offset		|		2		|	uint16_t	|	40			|
		|	18		|	Data Offset				|		2		|	uint16_t	|	60			|
		|	20		|	Data Compressed			|		1		|	uint8_t		|	1			|
		|	21		|	Data Element Size (B)	|		1		|	uint8_t		|	1			|
		|	25		|	Data Compressed Size (B)|		4		|	uint32_t	|	D			|
		|	29		|	Number of Nodes			|		4		|	int32_t		|	N			|
		|	33		|	Number of Time Steps	|		4		|	uint32_t	|	T			|
		|	37		|	Quench Type				|		1		|	uint8_t		|	Q			|
		|	38		|	Padding					|		2		|	char[2]		|				|
		|---------------------------------------------------------------------------------------|
		|---------------------------------------------------------------------------------------|
		|---------------------------------------------------------------------------------------|
		|										 PARAMETERS										|
		|---------------------------------------------------------------------------------------|
		|	OFFSET	|		DATA FIELD			|	SIZE (B)	|	TYPE		|	VALUE		|
		|---------------------------------------------------------------------------------------|
		|											Q == 0 (NONE)								|
		|---------------------------------------------------------------------------------------|
		|	40		|	Temperature				|		4		|	float		|		A		|
		|	44		|	Padding					|		16		|	char[16]	|				|
		|---------------------------------------------------------------------------------------|
		|											Q == 1 (FREEZE)								|
		|---------------------------------------------------------------------------------------|
		|	40		|	Initial Temperature		|		4		|	float		|		A		|
		|	44		|	Quench Temperature		|		4		|	float		|		q		|
		|	48		|	Padding					|		12		|	char[12]	|				|
		|---------------------------------------------------------------------------------------|
		|											Q == 2 (LINEAR)								|
		|---------------------------------------------------------------------------------------|
		|	40		|	Initial Temperature		|		4		|	float		|		A		|
		|	44		|	Gradient				|		4		|	float		|		B		|
		|	48		|	Padding					|		12		|	char[12]	|				|
		|---------------------------------------------------------------------------------------|
		|											Q == 3 (EXPONENTIAL)						|
		|---------------------------------------------------------------------------------------|
		|	40		|	Initial Temperature		|		4		|	float		|		A		|
		|	44		|	Exponential Gradient	|		4		|	float		|		B		|
		|	48		|	Padding					|		12		|	char[12]	|				|
		|---------------------------------------------------------------------------------------|
		|											Q == 4 (STEP)								|
		|---------------------------------------------------------------------------------------|
		|	40		|	Initial Temperature		|		4		|	float		|		A		|
		|	44		|	Step Gradient			|		4		|	float		|		B		|
		|	48		|	Step Period				|		4		|	uint32_t	|		P		|
		|	52		|	Padding					|		8		|	char[8]		|				|
		|---------------------------------------------------------------------------------------|
		|---------------------------------------------------------------------------------------|
		|---------------------------------------------------------------------------------------|
		|	OFFSET	|		DATA FIELD			|	SIZE (B)	|	TYPE		|	VALUE		|
		|---------------------------------------------------------------------------------------|
		|	60		|	Compressed Spin Data	|	   D		|				|				|
		|---------------------------------------------------------------------------------------|

		//////////////////////////////////////NOTES//////////////////////////////////////////////

		- D will be dependent on how well zlib was able to compress the data.
		- The step factor 'B' is a

		*////////////////////////////////////////////////////////////////////////////////////////

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

			void save( const std::string dirname, const std::string filename );
		};

		class IsingSpinModel_parallel : public IsingSpinModelBase
		{
		public:
			IsingSpinModel_parallel( Graph::GraphShared &graphshared, const std::uint32_t run_time, const std::uint32_t seed ) : IsingSpinModelBase( graphshared, run_time, seed ) {};
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

			void save( const std::string dirname, const std::string filename );
		};

		namespace detail
		{
#define BOLTZMANN_CONSTANT 1.38064852e-23
#define MAJOR_VERSION 1
#define MINOR_VERSION 0

			double calculate_beta( const double temperature )
			{
				return ( ( double ) 1. / ( BOLTZMANN_CONSTANT * temperature ) );
			}

			void save( const std::string dirname, const std::string filename, Matrix<std::uint8_t> &spin_states )
			{			
				std::string		fileid("xSPINSTATEx");
				std::uint8_t	fileversion[ 2 ] = { MAJOR_VERSION, MINOR_VERSION };
				std::uint8_t	zlibversion[ 2 ] = { 1, 0 };
				std::uint16_t	parameter_offset = 40;
				std::uint16_t	data_offset = 60;
				std::uint8_t	data_compressed = 0;
				std::uint8_t	data_element_size = 1;
				std::uint32_t	data_compressed_size = spin_states.length();
				std::uint32_t	number_of_nodes = spin_states.number_of_rows();
				std::uint32_t	number_of_time_steps = spin_states.number_of_columns();
				std::uint8_t	quench_type = 0;

				std::string filepath = dirname + "/" + filename;
				FileHandle file( filepath, "wb" );
			}


			__device__ double generateIID( curandState* globalState, int n )
			{
				//Generate a random number between 0 and 1.
				curandState localState = globalState[ n ];
				double RANDOM = curand_uniform_double( &localState );
				globalState[ n ] = localState;

				return RANDOM;
			}

			__device__ std::size_t countNeighbourMatches( const Graph::Graph* graph, const Matrix<std::int8_t>* spin_states, const std::size_t time_step, const int n )
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

			__device__ double calcFlipProb( const Graph::Graph* graph, const Matrix<std::int8_t>* spin_states, const double beta, const std::size_t time_step, const int n )
			{
				//Calculate the probability of vertex n flipping state.
				int S = countNeighbourMatches( graph, spin_states, time_step, n );

				double numerator = std::exp( -2 * beta * ( 2 * S - graph->degree( n ) ) );
				double denominator = 1. + numerator;
			}
		}
	}
}

#endif