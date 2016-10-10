#include "Ising_Spin_Model.h"


using namespace kspace;
using namespace IsingSpinModel;
using namespace detail;


double IsingSpinModel_MetropolisHastings::calcAcceptanceProb( const std::int32_t node, const double beta, const std::uint32_t time_step ) const
{
	std::int32_t neighbour = -1;
	double H1( 0 ), H2( 0 );

	for ( std::size_t k = 0; k < _graph.degree( node ); ++k )
	{
		neighbour = _graph.neighbour( node, k );
		if ( _spin_states.get( time_step, node ) == _spin_states.get( time_step, neighbour ) )
		{
			//If the spin state of node and neighbour are the same during time_step then it reduces the energy of H1 by 1.
			H1--;
		}
		else
		{
			//If the spin state of node and neighbour are the not the same during time_Step then it reduces the energy of H2 by 1.
			//This is essentially the same as considering the case 
			H2--;
		}
	}

	const double dH = H2 - H1 ;
	if ( dH > 0 )
	{
		return std::exp( -beta *  dH );
	}
	else
	{
		return 1.;
	}
}

void IsingSpinModel_MetropolisHastings::flip( const std::uint32_t time_step, const std::int32_t node )
{
	std::int8_t s = _spin_states.get( time_step, node );
	_spin_states.set( time_step, node, -s );
}

void IsingSpinModel_MetropolisHastings::run( const std::uint32_t time_step, const std::uint32_t temperature )
{
	double beta = calculate_beta( temperature );

	/*//////////////////////////////////////////////////
	/////////////////Procedure//////////////////////////
	////////////////////////////////////////////////////
		- Select a node N uniformly at random.
		- Calculate the Hamiltonians for flipping
		  and not flipping state.
		- Calculate the probability A of accepting
		  a change in state.
		- Generate a random number R uniformly at
		  random.
		- Switch state if R <= A.
	*///////////////////////////////////////////////////

	std::uniform_int_distribution<std::int32_t> uid( 0, _graph.number_of_nodes() - 1 );
	std::uniform_real_distribution<double> urd( 0, _graph.number_of_nodes() - 1 );

	std::int32_t node = uid( _mt_node_selector );
	double r = urd( _mt_flip_acceptor );
	memcpy( _spin_states.raw_data(time_step+1), _spin_states.raw_data(time_step), sizeof( std::int8_t ) * _spin_states.number_of_rows() );
	if ( r < calcAcceptanceProb( node, beta, time_step ) )
	{
		flip( time_step + 1, node );
	}
}