#include "Graph.h"
#include <stdexcept>

using namespace kspace;
using namespace Graph;
using namespace Parameters;

Type parameters_t::type() const
{
	return _type;
}

uint32_t parameters_t::numOfNodes() const
{
	return _numOfNodes;
}


///////////////////////////////////////////////////////
///////////////Rectangular Lattice\\\\\\\\\\\\\\\\\\\\\

uint32_t Rectangular_Lattice::width() const
{
	return _width;
}

uint32_t Rectangular_Lattice::height() const
{
	return _height;
}


///////////////////////////////////////////////////////
/////////////////Circular Lattice\\\\\\\\\\\\\\\\\\\\\\

Circular_Lattice::Circular_Lattice( const uint32_t numOfNodes, const uint32_t numOfDegreesPerNode ) : parameters_t( Type::Circular_Lattice, numOfNodes )
{
	if ( numOfDegreesPerNode >= numOfNodes )
	{
		throw std::invalid_argument( "number of degrees per node is greater than the number of nodes" );
	}

	if ( numOfDegreesPerNode % 2 != 0 )
	{
		throw std::invalid_argument( "number of degrees per node is not an even integer" );
	}

	_numOfDegreesPerNode = numOfDegreesPerNode;
}

uint32_t Circular_Lattice::numOfDegreesPerNode() const
{
	return _numOfDegreesPerNode;
}


////////////////////////////////////////////////////////
////////////////////Erdos Renyi\\\\\\\\\\\\\\\\\\\\\\\\\

double Erdos_Renyi::wiringProbability() const
{
	return _wiringProbability;
}

uint32_t Erdos_Renyi::seed() const
{
	return _seed;
}


///////////////////////////////////////////////////////
//////////////////Watts Strogatz\\\\\\\\\\\\\\\\\\\\\\\

Watts_Strogatz::Watts_Strogatz( const uint32_t numOfNodes, const uint32_t numOfDegreesPerNode, const double rewiringProbability, const uint32_t seed ) : parameters_t( Type::Watts_Strogatz, numOfNodes )
{
	if ( numOfDegreesPerNode >= numOfNodes )
	{
		throw std::invalid_argument( "number of degrees per node is greater than the number of nodes" );
	}

	if ( numOfDegreesPerNode % 2 != 0 )
	{
		throw std::invalid_argument( "number of degrees per node is not an even integer" );
	}

	_numOfDegreesPerNode = numOfDegreesPerNode;
	_rewiringProbability = rewiringProbability;
	_seed = seed;
}

uint32_t Watts_Strogatz::numOfDegreesPerNode() const
{
	return _numOfDegreesPerNode;
}

double Watts_Strogatz::rewiringProbability() const
{
	return _rewiringProbability;
}

uint32_t Watts_Strogatz::seed() const
{
	return _seed;
}


///////////////////////////////////////////////////////
/////////////////Barabasi Albert\\\\\\\\\\\\\\\\\\\\\\\

Barabasi_Albert::Barabasi_Albert( const uint32_t finNumOfNodes, const uint32_t initNumOfNodes, const uint32_t numOfDegreesPerNode, const uint32_t seed ) : parameters_t( Type::Barabasi_Albert, finNumOfNodes )
{
	if ( finNumOfNodes < initNumOfNodes )
	{
		throw std::invalid_argument( "the final number of nodes is less than initial number of nodes" );
	}

	if ( numOfDegreesPerNode > initNumOfNodes )
	{
		throw std::invalid_argument( "the number of degrees per node is greater than the initial number of nodes" );
	}

	_initNumOfNodes = initNumOfNodes;
	_numOfDegreesPerNode = numOfDegreesPerNode;
	_seed = seed;
}

uint32_t Barabasi_Albert::initNumOfNodes() const
{
	return _initNumOfNodes;
}

uint32_t Barabasi_Albert::numOfDegreesPerNode() const
{
	return _numOfDegreesPerNode;
}

uint32_t Barabasi_Albert::seed() const
{
	return _seed;
}