#ifndef GRAPH_H
#define GRAPH_H
#include "DataStructures.h"
#include "File_IO.h"
#include <cstdint>
#include <cassert>
#include <string>
#include <sstream>
#include <map>

/*///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////GRAPH FILE STRUCTURE//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

|-------------------------------------------------------------------------------------------|
|	  OFFSET	|		HEADER FIELD		|	SIZE (B)	|	TYPE		|	VALUE		|
|-------------------------------------------------------------------------------------------|
|		0		|	FILE ID					|		8		|	char		|	xKGRAPHx	|
|		8		|	Major Version			|		1		|	uint8_t		|	1			|
|		9		|	Minor Version			|		1		|	uint8_t		|	0			|
|		10		|	Parameter Offset		|		2		|	uint16_t	|	32			|
|		12		|	Data Offset				|		2		|	uint16_t	|	50			|
|		14		|	Data Size (B)			|		4		|	uint32_t	|	D=4(N+2M+Z)	|
|		18		|	Graph Type				|		2		|	uint16_t	|	G			|
|		20		|	Number of Nodes in Graph|		4		|	int32_t		|	N			|
|		24		|	Number of Nodes in File	|		4		|	int32_t		|	M			|
|		28		|	Number of Neighbours	|		4		|	uint32_t	|	Z			|
|-------------------------------------------------------------------------------------------|
|				The rest of the header depends on the value of Graph Type					|
|-------------------------------------------------------------------------------------------|
|									 Rectangular Lattice									|
|-------------------------------------------------------------------------------------------|
|		32		|	Width					|		4		|	int32_t		|				|
|		36		|	Height					|		4		|	int32_t		|				|
|		40		|	Padding					|		10		|	char[10]	|				|
|-------------------------------------------------------------------------------------------|
|									   Circular Lattice										|
|-------------------------------------------------------------------------------------------|
|		32		|	Degree Per Nodes		|		4		|	int32_t		|	0<=K<=N		|
|		36		|	Padding					|		14		|	char[14]	|				|
|-------------------------------------------------------------------------------------------|
|										  Erdos-Renyi										|
|-------------------------------------------------------------------------------------------|
|		32		|	Seed					|		4		|	uint32_t	|				|
|		36		|	Wiring Probability		|		4		|	float		|	0<=p_w<=1	|
|		40		|	Padding					|		10		|	char[10]	|				|
|-------------------------------------------------------------------------------------------|
|										Watts-Strogatz										|
|-------------------------------------------------------------------------------------------|
|		32		|	Seed					|		4		|	uint32_t	|				|
|		36		|	Degree Per Nodes		|		4		|	int32_t		|	0<=K<=N		|
|		40		|	Wiring Probability		|		4		|	float		|	0<=p_w<=1	|
|		44		|	Padding					|		6		|	char[6]		|				|
|-------------------------------------------------------------------------------------------|
|										Barabasi-Albert										|
|-------------------------------------------------------------------------------------------|
|		32		|	Seed					|		4		|	uint32_t	|				|
|		36		|	Initial Number of Nodes	|		4		|	int32_t		|		M		|
|		40		|	Degree Per Nodes		|		4		|	int32_t		|	 0<=K<=M	|
|		44		|	Padding					|		6		|	char[6]		|				|
|-------------------------------------------------------------------------------------------|
|-------------------------------------------------------------------------------------------|
|-------------------------------------------------------------------------------------------|
|	OFFSET		|		DATA FIELD			|	SIZE (B)	|	TYPE		|	VALUE		|
|-------------------------------------------------------------------------------------------|
| 50			|	Degrees					|	   4N		|	uint32_t[N]	|	uint32_t	|
| 50+4N			|	Vertex IDs				|	   4M		|	int32_t[M]	|	int32_t		|
| 50+4(N+M)		|	Neighbour Offsets		|	   4(M+1)	| uint32_t[M+1]	|	uint32_t	|
| 50+4(N+2M+1)	|	Neighbour IDs			|	   4Z		|	int32_t[Z]	|	int32_t		|
|-------------------------------------------------------------------------------------------|

//////////////////////////////////////////NOTES//////////////////////////////////////////////

	- Both directed and undirected graphs can be stored.
	- The values of M and N is dependent on the graph.
	- Neighbour IDs is a jagged list of neighbours.
	- Directed edges are indicated by a negative Neighbour ID. Undirected edges
	  are indicated by positive Neighbour IDs.
	- Each edge appears only once. Given an edge {n,m} then for Node ID n the
	  Neighbour ID 'm' only appears if m>=n.

*////////////////////////////////////////////////////////////////////////////////////////////

namespace kspace
{
	namespace Graph
	{

#define MAJOR_VERSION 1
#define MINOR_VERSION 0

		enum GraphType : std::uint16_t
		{
			NONE, LINEAR_LATTICE, RECTANGULAR_LATTICE, CIRCULAR_LATTICE, ERDOS_RENYI, WATTS_STROGATZ, BARABASI_ALBERT
		};

		namespace Parameters
		{
			struct Rectangular_Lattice
			{
				std::int32_t number_of_nodes;
				std::int32_t width;
				std::int32_t height;

				void operator()( const std::int32_t width, const std::int32_t height );
				void clear() { ( *this )( NULL, NULL ); }

				Rectangular_Lattice() { clear(); }
				Rectangular_Lattice( const std::int32_t width, const std::int32_t height ) { ( *this )( width, height ); }
			};

			struct Circular_Lattice
			{
				std::int32_t number_of_nodes;
				std::int32_t number_of_degrees;

				void operator()( const std::int32_t number_of_nodes, const std::int32_t number_of_degrees );
				void clear() { ( *this )( NULL, NULL ); }

				Circular_Lattice() { clear(); }
				Circular_Lattice( const std::int32_t number_of_nodes, const std::int32_t number_of_degrees ) { ( *this )( number_of_nodes, number_of_degrees ); }
			};

			struct Erdos_Renyi
			{
				std::int32_t number_of_nodes;
				float wiring_probability;
				std::uint32_t seed;

				void operator()( const std::int32_t number_of_nodes, const float wiring_probability, const uint32_t seed );
				void clear() { ( *this )( NULL, NULL, NULL ); }
				Erdos_Renyi() { clear(); }
				Erdos_Renyi( const std::int32_t number_of_nodes, const float wiring_probability, const uint32_t seed ) { ( *this )( number_of_nodes, wiring_probability, seed ); }
			};

			struct Watts_Strogatz
			{
				std::int32_t number_of_nodes;
				std::int32_t number_of_degrees;
				float rewiring_probability;
				std::uint32_t seed;

				void operator()( const std::int32_t number_of_nodes, const std::int32_t number_of_degrees, const float rewiring_probability, const uint32_t seed );
				void clear() { ( *this )( NULL, NULL, NULL, NULL ); }

				Watts_Strogatz() { clear(); }
				Watts_Strogatz( const std::int32_t number_of_nodes, const std::int32_t number_of_degrees, const float rewiring_probability, const uint32_t seed ) { ( *this )( number_of_nodes, number_of_degrees, rewiring_probability, seed ); }
			};

			struct Barabasi_Albert
			{
				std::int32_t final_number_of_nodes;
				std::int32_t initial_number_of_nodes;
				std::int32_t number_of_degrees;
				std::uint32_t seed;

				void operator()( const std::int32_t init_number_of_nodes, const std::int32_t final_number_of_nodes, const std::int32_t number_of_degrees, const uint32_t seed );
				void clear() { ( *this )( NULL, NULL, NULL, NULL ); }

				Barabasi_Albert() { clear(); }
				Barabasi_Albert( const std::int32_t init_number_of_nodes, const std::int32_t final_number_of_nodes, const std::int32_t number_of_degrees, const uint32_t seed ) { ( *this )( init_number_of_nodes, final_number_of_nodes, number_of_degrees, seed ); }
			};

			class Graph_Parameters
			{
			private:

			public:
				GraphType type;
				Rectangular_Lattice rectangular_lattice;
				Circular_Lattice circular_lattice;
				Erdos_Renyi erdos_renyi;
				Watts_Strogatz watts_strogatz;
				Barabasi_Albert barabasi_albert;

				Graph_Parameters( const std::string filename );
				Graph_Parameters()									{ clear(); type = NONE; }
				Graph_Parameters( const Rectangular_Lattice& rl )	{ clear(); rectangular_lattice = rl; };
				Graph_Parameters( const Circular_Lattice& cl )		{ clear(); circular_lattice = cl; }
				Graph_Parameters( const Erdos_Renyi& er )			{ clear(); erdos_renyi = er; }
				Graph_Parameters( const Watts_Strogatz& ws )		{ clear(); watts_strogatz = ws; }
				Graph_Parameters( const Barabasi_Albert& ba )		{ clear(); barabasi_albert = ba; }

				void clear()
				{
					type = NONE;
					rectangular_lattice.clear();
					circular_lattice.clear();
					erdos_renyi.clear();
					watts_strogatz.clear();
					barabasi_albert.clear();
				}
			};
		};

		class Graph
		{
			friend class GraphShared;
		private:
			MemoryLocation _memloc;

			std::int32_t* _number_of_nodes;

			std::uint8_t* _adjmat;
			std::int32_t* _adjlist;
			std::int32_t* _degrees;
			std::uint32_t* _offsets;
		public:
			Graph( const std::string fname, const MemoryLocation memloc );
			~Graph();

			int32_t number_of_nodes() const;
			int32_t degree( const uint32_t v ) const;

			bool is_connected( const uint32_t v, const uint32_t w ) const;
			int32_t neighbour( const uint32_t v, const uint32_t kth_neighbour ) const;

			MemoryLocation memory_location() const;
		};

		class GraphShared
		{
			friend class GraphShared;
		private:
			Graph* intermediary;

		public:
			Graph* host;
			Graph* device;


			GraphShared( const std::string filename );
			~GraphShared();

			//Remove the default copy constructors
			GraphShared( const FILEIO::FileHandle& ) = delete;
			GraphShared& operator=( const GraphShared& ) = delete;

			//Define move constructors
			GraphShared( GraphShared& other );
			GraphShared& operator=( GraphShared&& rhs );

			void host2device();
			void device2host();
		};
	}
}

#endif