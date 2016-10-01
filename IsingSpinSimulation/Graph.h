#ifndef GRAPH_H
#define GRAPH_H
#include "DataStructures.h"
#include "File_IO.h"
#include <cstdint>
#include <cassert>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include "KDetails.h"

/*/////////////////////////////////////////////////|\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
||||||||||||||||||||||||||||||||||||||||| Graph File Structure |||||||||||||||||||||||||||||||||||||||||
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\|//////////////////////////////////////////////////

|-----------------------------------------------------------------------------------------------|
|	  OFFSET	|		HEADER FIELD		|	SIZE (B)	|	  TYPE		|	    VALUE		|
|-----------------------------------------------------------------------------------------------|
|		0		| FILE ID					|		8		|	char		| mKGRAPHm/eKGRAPHe	|
|		8		| Major Version				|		1		|	uint8_t		|		  1			|
|		9		| Minor Version				|		1		|	uint8_t		|		  1			|
|		10		| Parameter Offset			|		2		|	uint16_t	|		  50		|
|		12		| Number of Parameters		|		1		|	uint8_t		|		  P			|
|		13		| P. Uncompressed Size (B)	|		4		|	uint32_t	|		 PUS		|
|		17		| P. Cmprssd Sz				|		4		|	uint32_t	|		 PCS		|
|		21		| Data Offset				|		2		|	uint16_t	|	      DO		|
|		23		| D. Uncompressed Size (B)	|		4		|	uint32_t	|		  U			|
|		27		| D. Bitpacked Size (B)		|		4		|	uint32_t	|		  B			|
|		31		| D. Compressed Size (B)	|		4		|	uint32_t	|		  C			|
|		35		| Number of Nodes in Graph	|		4		|	int32_t		|		  N			|
|		39		| Number of Nodes in File	|		4		|	int32_t		|		  M			|
|		43		| Number of Neighbours		|		4		|	uint32_t	|		  Z			|
|		47		| Graph ID					|		1		|	uint8_t		|		  G			|
|-----------------------------------------------------------------------------------------------|
|-----------------------------------------------------------------------------------------------|
|-----------------------------------------------------------------------------------------------|
|		  The rest of the header depends on the number of parameters and the sizes of			|
|							   the array of names and values.									|
|-----------------------------------------------------------------------------------------------|
|		50		|  P. Names					|		PNS		|	char[]		|					|
|				|  P. Types					|		PTS		|	uint8_t[]	|					|
|				|  P. Sizes					|				|	uint8_t[]	|					|
|				|  P. Values				|		PVS		|	variable[]	|					|
|-----------------------------------------------------------------------------------------------|
|-----------------------------------------------------------------------------------------------|
|-----------------------------------------------------------------------------------------------|
|											Matrix Data Format									|
|-----------------------------------------------------------------------------------------------|
|	  OFFSET	|		DATA FIELD			|	SIZE (B)	|	  TYPE		|	    VALUE		|
|-----------------------------------------------------------------------------------------------|
|       DO		|  Matrix					|		C		|	 Byte[C]	|	   uint8_t		|
|-----------------------------------------------------------------------------------------------|
|										  Edge List Data Format									|
|-----------------------------------------------------------------------------------------------|
|	  OFFSET	|		DATA FIELD			|	SIZE (B)	|	  TYPE		|		VALUE		|
|-----------------------------------------------------------------------------------------------|
|       DO		|  Degrees					|				|				|	   uint32_t		|
|				|  Vertex IDs				|				|				|	   int32_t		|
|				|  Neighbour Offsets		|				|				|	   uint32_t		|
|				|  Neighbour IDs			|				|				|	   int32_t		|
|-----------------------------------------------------------------------------------------------|

////////////////////////////////////////////|\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
||||||||||||||||||||||||||||||||||        NOTES        |||||||||||||||||||||||||||||||||||||||
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\|////////////////////////////////////////////////

- Both directed and undirected graphs can be stored.
- The values of M and N is dependent on the graph.
- Neighbour IDs is a jagged list of neighbours.
- Directed edges are indicated by a negative Neighbour ID. Undirected edges
are indicated by positive Neighbour IDs.
- Each edge appears only once. Given an edge {n,m} then for Node ID n the
Neighbour ID 'm' only appears if m >= n.

///////////////////////////////////////////|\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\|//////////////////////////////////////////////*/

namespace kspace
{
	namespace GRAPH
	{

#define _MAJOR_VERSION_ 1
#define _MINOR_VERSION_ 0
		enum class ID : std::uint16_t
		{
			NONE, LINEAR_LATTICE, RECTANGULAR_LATTICE, CIRCULAR_LATTICE, ERDOS_RENYI, WATTS_STROGATZ, BARABASI_ALBERT
		};


		class Parameters;

		
		class Graph
		{
			friend class GraphShared;
		private:
			MemoryLocation memloc;

			std::int32_t* number_of_nodes;

			std::uint8_t*	adjmat;
			std::int32_t*	adjlist;
			std::int32_t*	degrees;
			std::uint32_t*	offsets;

			class GET : public GET_SUPER < Graph >
			{
			public:
				GET( const Graph &parent ) : GET_SUPER::GET_SUPER( parent ) {};
				int32_t number_of_nodes() const;
				int32_t degree( const uint32_t v ) const;

				bool is_connected( const uint32_t v, const uint32_t w ) const;
				int32_t neighbour( const uint32_t v, const uint32_t kth_neighbour ) const;

				MemoryLocation memory_location() const;
			};
		protected:
			struct Data {
				std::int32_t* num_of_nodes;
				std::uint8_t* adjmat;
				std::int32_t* adjlist;
				std::int32_t* degrees;
				std::uint32_t* offsets;
			};

			Data readData( const FILEIO::FileHandle &file );
		public:
			Graph( const std::string fname, const MemoryLocation memloc );
			~Graph();

			GET get;
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