#ifndef GRAPH_H
#define GRAPH_H
#include "details.h"
#include "File_IO.h"
#include <cstdint>
#include <cassert>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include "details.h"

/*/////////////////////////////////////////////////|\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
||||||||||||||||||||||||||||||||||||||||| Graph File Structure |||||||||||||||||||||||||||||||||||||||||
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\|//////////////////////////////////////////////////

|-----------------------------------------------------------------------------------------------|
|	  OFFSET	|		HEADER FIELD		|	SIZE (B)	|	  TYPE		|	    VALUE		|
|-----------------------------------------------------------------------------------------------|
|		0		| FILE ID					|		8		|	char		|	   xKGRAPHx		|
|		8		| Major Version				|		1		|	uint8_t		|		  1			|
|		9		| Minor Version				|		1		|	uint8_t		|		  1			|
|		10		| Parameter Offset			|		2		|	uint16_t	|		  44		|
|		12		| Number of Parameters		|		1		|	uint8_t		|		  P			|
|		13		| P. Uncompressed Size (B)	|		4		|	uint32_t	|		 PUS		|
|		17		| P. Cmprssd Sz				|		4		|	uint32_t	|		 PCS		|
|		21		| Data Offset				|		2		|	uint16_t	|	      DO		|
|		23		| D. Uncompressed Size (B)	|		4		|	uint32_t	|		  U			|
|		27		| D. Bitpacked Size (B)		|		4		|	uint32_t	|		  B			|
|		31		| D. Compressed Size (B)	|		4		|	uint32_t	|		  C			|
|		35		| Number of Nodes in Graph	|		4		|	int32_t		|		  N			|
|		39		| Number of Edges in Graph	|		4		|	uint32_t	|		  K			|
|		43		| Graph ID					|		1		|	uint8_t		|		  G			|
|-----------------------------------------------------------------------------------------------|
|-----------------------------------------------------------------------------------------------|
|-----------------------------------------------------------------------------------------------|
|		  The rest of the header depends on the number of parameters and the sizes of			|
|							   the array of names and values.									|
|-----------------------------------------------------------------------------------------------|
|		44		|  P. Names					|		PNS		|	char[]		|					|
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

#define _MAJOR_VERSION_ 1
#define _MINOR_VERSION_ 0

	namespace GRAPH
	{
		class Parameters;

		/**
		* A data structure for storing graphs of arbitrary type and topology on the device or host memory.
		* Can load graphs from file of the .kgraph.
		*/
		class Graph
		{
			friend class GraphShared;
		protected:
			/**
			* Contains the resources required for representing the graph.
			* We use this method as it is useful when loading from file
			* when constructing graphs from scratch.
			*/
			class Data
			{
			private:
				/**
				* Helper function that moves the resources managed by the passed Data object to the one calling the function.
				*/
				void move_data( Data&& that );
			public:
				MemoryLocation memloc;			/**< Indicates the memory location of the data being managed. */
				std::int32_t* number_of_nodes;	/**< A pointer the number of nodes in the graph. */
				std::uint8_t* adjmat;			/**< A 1D representation of an adjacency matrix. */
				std::int32_t* adjlist;			/**< A 1D representation of an adjacency list. */
				std::int32_t* degrees;			/**< A 1D array containing the number of degrees of each node. */
				std::uint32_t* offsets;			/**< A 1D array containing the offset for the beginning of the neighbour list for each node in the adjacency list.*/

				/**
				* Constructs an empty container by default.
				*/
				Data() : number_of_nodes( nullptr ), adjmat( nullptr ), adjlist( nullptr ), degrees( nullptr ), offsets( nullptr ), memloc( MemoryLocation::host ) {};

				~Data(); 

				Data( Data const  & ) = delete; /**< Delete the copy constructor.*/
				Data& operator=( Data const & ) = delete; /**< Delete the copy assignment operator. */

				Data( Data&& that );
				Data& operator=( Data&& that );

				void clear();
			} data;

			/**
			* Contains methods which provide read only access the data being stored in the graph.
			*/
			class GRAPH_GET
			{
				Graph const & parent; /**< A reference to the Graph object to which private members are accessible. */
			public:
				/**< On construction the Graph to which private member access is available is set. */
				GRAPH_GET( const Graph& parent ) : parent( parent ) {};

				int32_t const & number_of_nodes() const; 
				int32_t const & degree( const uint32_t v ) const; 
				std::uint32_t const & offset(const size_t v) const;

				bool const & is_connected( const uint32_t v, const uint32_t w ) const;
				int32_t const & neighbour(const uint32_t v, const uint32_t kth_neighbour) const;

				MemoryLocation const & memory_location() const;

				std::uint8_t const * adjmat() const;
				std::int32_t const * adjlist() const;
				std::int32_t const * degrees() const;
				std::uint32_t const * offsets() const;
			};

			/**
			* A container of methods which provide read and write access to the data stored in the Graph.
			*/
			class GRAPH_SET
			{
				Graph& parent;
			public:
				/**< On construction the Graph to which private member access is available is set. */
				GRAPH_SET( Graph& parent ) : parent( parent ) {};

				std::uint8_t* adjmat() const;
				std::int32_t* adjlist() const;
				std::int32_t* degrees() const;
				std::uint32_t* offsets() const;
			};

			Graph::Data&& readData( const FILEIO::FileHandle &file );
			void initialise( Data& data, MemoryLocation const & memloc );
		public:

			/**
			* An enum class the elements of which indicate the type of a graph stored in a Graph object.
			*/
			enum class ID : std::uint8_t
			{
				NONE, LINEAR_LATTICE, RECTANGULAR_LATTICE, CIRCULAR_LATTICE, ERDOS_RENYI, WATTS_STROGATZ, BARABASI_ALBERT
			};

			GRAPH_GET get;
			GRAPH_SET set;

			Graph();

			
			Graph( std::string const & fname, MemoryLocation const & memloc );
			Graph( Data& data, MemoryLocation const & memloc );
		};
	}
}

#endif