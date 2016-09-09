#ifndef GRAPH_H
#define GRAPH_H
#include "DataStructures.h"
#include "File_IO.h"
#include <cstdint>
#include <cassert>
#include <string>
#include <sstream>
#include <map>

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
		enum Type : std::uint32_t
		{
			RECTANGULAR_LATTICE, CIRCULAR_LATTICE, ERDOS_RENYI, WATTS_STROGATZ, BARABASI_ALBERT
		};

		typedef std::string parameter_t;

		class Parameters
		{
		public:
			static class cast {
			public:
				typedef std::int32_t number_of_nodes_t, width_t, height_t, number_of_degrees_t, initial_number_of_nodes_t;
				typedef float wiring_probability_t, rewiring_probability_t;
			};

			enum class key { TYPE, NUMBER_OF_NODES, WIDTH, HEIGHT, NUMBER_OF_DEGREES, WIRING_PROBABILITY, REWIRING_PROBABILITY, INITIAL_NUMBER_OF_NODES };

			Parameters(const Type type, const std::int32_t number_of_nodes, const std::int32_t width, const std::int32_t height);
			Parameters(const Type type, const std::int32_t number_of_nodes, const std::int32_t number_of_degrees);
			Parameters(const Type type, const std::int32_t number_of_nodes, const float wiring_probability);
			Parameters(const Type type, const std::int32_t number_of_nodes, const std::int32_t number_of_degrees, const float rewiring_probability);
			Parameters(const Type type, const std::int32_t number_of_nodes, const std::int32_t initial_number_of_nodes, const std::int32_t number_of_degrees);

			parameter_t operator()(const key parameter) const;

			void read(FileHandle &file);
			void write(FileHandle &file);
		private:
			Type _type;
			std::int32_t _numOfNodes;
			std::map<key, std::string> _parameters;
		} parameters;

		template <class T> T parameter_cast(const parameter_t &parameter_value)
		{
			std::stringstream ss;
			ss << parameter_value;

			T value;
			ss >> value;

			return value;
		}

		template <class T> parameter_t parameter_cast(const T &parameter_value)
		{
			return std::to_string(parameter_value);
		}

		struct GraphHeader
		{
			char file_type[8];
			std::uint8_t version[2];
			std::uint16_t data_offset;
			std::uint32_t data_size;
			std::uint16_t graph_type;
			std::uint32_t num_of_nodes_in_graph;
			std::uint32_t num_of_nodes_in_file;
			std::uint32_t num_of_neighbours;
			Parameters parameters;
		};

		struct GraphData
		{
			std::uint32_t *degrees;
			std::int32_t *vertex_ids;
			std::uint32_t *neighbour_offsets;
			std::int32_t *neighbour_ids;
		};

		class Graph
		{
		private:
			MemoryLocation *_memLoc;

			std::uint8_t *_adjmat;
			std::uint32_t *_adjlist;
			std::uint32_t *_degrees;
			std::uint32_t *_offsets;

		protected:
			GraphHeader readHeader(FileHandle &file);
			GraphData readData(const FileHandle &file, const GraphHeader &hdr);

		public:

			Parameters *parameters;

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