#ifndef GRAPH_FILE_HEADER_H
#define GRAPH_FILE_HEADER_H

#include <cstdint>
#include "File_IO.h"
#include "Graph.h"

namespace kspace
{
	namespace GRAPH
	{
		class Header {
		private:
			kspace::FILEIO::Byte header[ 70 ];
		protected:
			void init();
			void clear_parameters();

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
				GRAPH::ID id() const;
				std::int32_t number_of_nodes() const;
				std::int32_t number_of_nodes_in_file() const;
				std::uint32_t number_of_neighbour_in_file() const;
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
				void id( const ID );
				void number_of_nodes_in_graph( const std::int32_t );
				void number_of_nodes_in_file( const std::int32_t );
				void number_of_neighbours_in_file( const std::uint32_t );
			};

		public:

			enum
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
				NUMBER_OF_NODES_IN_FILE = 39,
				NUMBER_OF_NEIGHBOURS_IN_FILE = 43,
				GRAPH_ID = 47
			};

			static struct
			{
				size_t FILE_ID = 8;
				size_t VERSION_MAJOR = 1;
				size_t VERSION_MINOR = 1;
				size_t OFFSET_PARAMETERS = 2;
				size_t NUM_OF_PARAMETERS = 1;
				size_t SIZE_P_UNCOMPRESSED = 4;
				size_t SIZE_P_COMPRESSED = 4;
				size_t OFFSET_DATA = 2;
				size_t SIZE_D_UNCOMPRESSED = 4;
				size_t SIZE_D_BITPACKED = 4;
				size_t SIZE_D_COMPRESSED = 4;
				size_t NUMBER_OF_NODES_IN_GRAPH = 4;
				size_t NUMBER_OF_NODES_IN_FILE = 4;
				size_t NUMBER_OF_NEIGHBOURS_IN_FILE = 4;
				size_t GRAPH_ID = 1;
			} field_size;

			Header();
			Header( Parameters &parameters );
			Header( const FILEIO::FileHandle& file );

			enum class GraphFileID { MATRIX, EDGE_LIST };

			GET get;
			SET set;
		};
	}
}
#endif