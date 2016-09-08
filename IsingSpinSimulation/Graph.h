#ifndef GRAPH_H
#define GRAPH_H
#include "DataStructures.h"
#include <cstdint>
#include <cassert>
#include <string>

/*///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////GRAPH FILE STRUCTURE///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

	|---------------------------------------------------------------------------------------|
	|	OFFSET	|		HEADER FIELD		|	SIZE (B)	|	TYPE		|	VALUE		|
	|---------------------------------------------------------------------------------------|
	|	0		|	FILE ID					|		8		|	char		|	xKGRAPHx	|
	|	8		|	Major Version			|		1		|	uint8_t		|	1			|
	|	9		|	Minor Version			|		1		|	uint8_t		|	0			|
	|	10		|	Data Offset				|		2		|	uint16_t	|	42			|
	|	12		|	Data Size (B)			|		4		|	uint32_t	|	D=4(N+2M+Z)	|
	|	16		|	Graph Type				|		2		|	uint16_t	|	G			|
	|	18		|	Number of Nodes in Graph|		4		|	uint32_t	|	N			|
	|	22		|	Number of Nodes in File	|		4		|	uint32_t	|	M			|
	|	26		|	Number of Neighbours	|		4		|	uint32_t	|	Z			|
	|---------------------------------------------------------------------------------------|
	|				The rest of the header depends on the value of Graph Type				|
	|---------------------------------------------------------------------------------------|
	|										Rectangular Lattice								|
	|---------------------------------------------------------------------------------------|
	|	30		|	Width					|		4		|	uint32_t	|				|
	|	34		|	Height					|		4		|	uint32_t	|				|
	|	38		|	Padding					|		4		|	char[4]		|				|
	|---------------------------------------------------------------------------------------|
	|										  Circular Lattice								|
	|---------------------------------------------------------------------------------------|
	|	30		|	Degree Per Nodes		|		4		|	uint32_t	|	0<=K<=N		|
	|	34		|	Padding					|		8		|	char[8]		|				|
	|---------------------------------------------------------------------------------------|
	|											Erdos-Renyi									|
	|---------------------------------------------------------------------------------------|
	|	30		|	Seed					|		4		|	uint32_t	|				|
	|	34		|	Wiring Probability		|		4		|	float		|	0<=p_w<=1	|
	|	38		|	Padding					|		4		|	char[4]		|				|
	|---------------------------------------------------------------------------------------|
	|										  Watts-Strogatz								|
	|---------------------------------------------------------------------------------------|
	|	30		|	Seed					|		4		|	uint32_t	|				|
	|	34		|	Degree Per Nodes		|		4		|	uint32_t	|	0<=K<=N		|
	|	38		|	Wiring Probability		|		4		|	float		|	0<=p_w<=1	|
	|---------------------------------------------------------------------------------------|
	|										  Barabasi-Albert								|
	|---------------------------------------------------------------------------------------|
	|	30		|	Seed					|		4		|	uint32_t	|				|
	|	34		|	Initial Number of Nodes	|		4		|	uint32_t	|		M		|
	|	38		|	Degree Per Nodes		|		4		|	uint32_t	|	0<=K<=M		|
	|---------------------------------------------------------------------------------------|
	|---------------------------------------------------------------------------------------|
	|---------------------------------------------------------------------------------------|
	|	OFFSET	|		DATA FIELD			|	SIZE (B)	|	TYPE		|	VALUE		|
	|---------------------------------------------------------------------------------------|
	| 42		|	Degrees					|	   4N		|	uint32_t[N]	|	uint32_t	|
	| 42+4N		|	Vertex IDs				|	   4M		|	int32_t[M]	|	int32_t		|
	| 42+4(N+M) |	Neighbour Offsets		|	   4M		|	uint32_t[M]	|	uint32_t	|
	| 42+4(N+2M)|	Neighbour IDs			|	   4Z		|	int32_t[Z]	|	int32_t		|
	|---------------------------------------------------------------------------------------|

//////////////////////////////////////NOTES//////////////////////////////////////////

		- Both directed and undirected graphs can be stored.
		- The values of M and N is dependent on the graph.
		- Neighbour IDs is a jagged list of neighbours.
		- Directed edges are indicated by a negative Neighbour ID. Undirected edges
		  are indicated by positive Neighbour IDs.
		- Each edge appears only once. Given an edge {n,m} then for Node ID n the
		  Neighbour ID 'm' only appears if m>=n.
		  

*////////////////////////////////////////////////////////////////////////////////////

namespace kspace
{
	namespace Graph
	{
		enum class Type : std::uint8_t
		{
			Rectangular_Lattice, Circular_Lattice, Erdos_Renyi, Watts_Strogatz, Barabasi_Albert
		};

		namespace Parameters
		{
			class parameters_t
			{
			private:
				Type _type;

			protected:
				uint32_t _numOfNodes;

			public:
				parameters_t(const Type type, const uint32_t numOfNodes) : _type(type), _numOfNodes(numOfNodes) {};

				Type type() const;
				uint32_t numOfNodes() const;
			};


			class Rectangular_Lattice : public parameters_t
			{
			private:
				uint32_t _width;
				uint32_t _height;

			public:
				Rectangular_Lattice::Rectangular_Lattice(const uint32_t width, const uint32_t height) : parameters_t(Type::Rectangular_Lattice, width * height), _width(width), _height(height) {};

				uint32_t width() const;
				uint32_t height() const;
			};


			class Circular_Lattice : public parameters_t
			{
			private:
				uint32_t _numOfDegreesPerNode;

			public:
				Circular_Lattice(const uint32_t numOfNodes, const uint32_t numOfDegreesPerNode);

				uint32_t numOfDegreesPerNode() const;
			};


			static struct Erdos_Renyi : public parameters_t
			{
			private:
				double _wiringProbability;
				uint32_t _seed;

			public:
				Erdos_Renyi(const uint32_t numOfnodes, const double wiringProbability, const uint32_t seed) : parameters_t(Type::Erdos_Renyi, numOfNodes), _wiringProbability(wiringProbability), _seed(seed) {};
				
				double wiringProbability() const;
				uint32_t seed() const;
			};


			class Watts_Strogatz : public parameters_t
			{
			private:
				uint32_t _numOfDegreesPerNode;
				double _rewiringProbability;
				uint32_t _seed;

			public:
				Watts_Strogatz(const uint32_t numOfNodes, const uint32_t numOfDegreesPerNode, const double rewiringProbability, const uint32_t seed);
				
				uint32_t numOfDegreesPerNode() const;
				double rewiringProbability() const;
				uint32_t seed() const;
			};


			class Barabasi_Albert : public parameters_t
			{
			private:
				uint32_t _initNumOfNodes;
				uint32_t _numOfDegreesPerNode;
				uint32_t _seed;
			public:
				Barabasi_Albert(const uint32_t finNumOfNodes, const uint32_t initNumOfNodes, const uint32_t numOfDegreesPerNode, const uint32_t seed);

				uint32_t initNumOfNodes() const;
				uint32_t numOfDegreesPerNode() const;
				uint32_t seed() const;
			};
		};

		class Graph
		{
		private:
			MemoryLocation *_memLoc;

			std::uint8_t *_adjmat;
			std::uint32_t *_adjlist;
			std::uint32_t *_degrees;
			std::uint32_t *_offsets;
		public:

			Parameters::parameters_t *parameters;

			Graph(const std::string fname, const MemoryLocation memloc);

			uint32_t numOfNodes() const;
			uint32_t degree(const uint32_t v) const;

			bool is_connected(const uint32_t v, const uint32_t w) const;
			uint32_t neighbour(const uint32_t v, const uint32_t kth_neighbour) const;

			MemoryLocation memory_location() const;
		};

		class GraphShared
		{
		private:
			Graph *intermediary;
		public:
			Graph *host;
			Graph *device;

			GraphShared(const std::string fname);
			~GraphShared();

			void host2device();
			void device2host();
		};
	}	
}

#endif