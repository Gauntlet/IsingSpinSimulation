#ifndef GRAPH_FILE_HEADER_H
#define GRAPH_FILE_HEADER_H

#include <cstdint>
#include "File_IO.h"
#include "Graph.h"
#include <zlib.h>

namespace kspace
{
	namespace GRAPH
	{
		class Header
		{
			private:
			static const size_t header_size = 44;
			Byte header[ header_size ];
			protected:
			void init();

			class GET : public GET_SUPER < Header >
			{
				public:
				GET( const Header& parent ) : GET_SUPER::GET_SUPER( parent ) {};

				std::uint8_t major() const;
				std::uint8_t minor() const;
				std::uint16_t offset_parameters() const;
				std::uint16_t offset_data() const;
				bool is_compressed() const;
				bool is_bitpacked() const;
				std::uint32_t size_parameters_uncompressed() const;
				std::uint32_t size_parameters_compressed() const;
				std::uint32_t size_data_uncompressed() const;
				std::uint32_t size_data_bitpacked() const;
				std::uint32_t size_data_compressed() const;
				std::int32_t number_of_nodes_in_graph() const;
				std::uint32_t number_of_edges_in_graph() const;
				Graph::ID id() const;
			};

			class SET : public SET_SUPER < Header >
			{
				public:
				SET( Header& parent ) : SET_SUPER::SET_SUPER( parent ) {};

				void major( const std::uint8_t );
				void minor( const std::uint8_t );
				void offset_data( const std::uint16_t );
				void size_parameters_uncompressed( const std::uint32_t );
				void size_parameters_compressed( const std::uint32_t );
				void size_data_uncompressed( const std::uint32_t );
				void size_data_bitpacked( const std::uint32_t );
				void size_data_compressed( const std::uint32_t );
				void number_of_nodes_in_graph( const std::int32_t );
				void number_of_edges_in_graph( const std::uint32_t );
				void id( const ID );
			};

			public:

			enum class OFFSET : size_t
			{
				FILE_ID = 0,
				VERSION_MAJOR = 8,
				VERSION_MINOR = 9,
				OFFSET_PARAMETERS = 10,
				NUM_OF_PARAMETERS = 12,
				SIZE_P_UNCOMPRESSED = 13,
				SIZE_P_COMPRESSED = 17,
				OFFSET_DATA = 21,
				SIZE_D_UNCOMPRESSED = 23,
				SIZE_D_BITPACKED = 27,
				SIZE_D_COMPRESSED = 31,
				NUMBER_OF_NODES_IN_GRAPH = 35,
				NUMBER_OF_EDGES_IN_GRAPH = 39,
				GRAPH_ID = 43
			};

			enum class FIELD_SIZE : size_t
			{
				FILE_ID = 8,
				VERSION_MAJOR = 1,
				VERSION_MINOR = 1,
				OFFSET_PARAMETERS = 2,
				NUM_OF_PARAMETERS = 1,
				SIZE_P_UNCOMPRESSED = 4,
				SIZE_P_COMPRESSED = 4,
				OFFSET_DATA = 2,
				SIZE_D_UNCOMPRESSED = 4,
				SIZE_D_BITPACKED = 4,
				SIZE_D_COMPRESSED = 4,
				NUMBER_OF_NODES_IN_GRAPH = 4,
				NUMBER_OF_EDGES_IN_GRAPH = 4,
				GRAPH_ID = 1
			} field_size;

			Header();
			Header( const FILEIO::FileHandle& file );

			enum class GraphFileID { MATRIX, EDGE_LIST };

			GET get;
			SET set;
		};
	}
}
#endif