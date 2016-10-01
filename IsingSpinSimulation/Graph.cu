#include "Graph.h"
#include <fstream>
#include "File_IO.h"
#include <map>
#include <memory>
#include "Graph_File_Header.h"

using namespace kspace::GRAPH;
using namespace kspace::FILEIO;



Graph::Data Graph::readData( const FILEIO::FileHandle &file)
{
	Data data = { nullptr, nullptr, nullptr, nullptr, nullptr };

	
	uint16_t data_offset(0);
	int seeksuccess = fseek( file(), Header::OFFSET_DATA, SEEK_SET );
	if ( !seeksuccess )
	{
		
		fread( &data_offset, Header::field_size.OFFSET_DATA, 1, file() );
	}
	else
	{
		throw std::runtime_error( "fseek to data offset failed: " + std::to_string( seeksuccess ) );
	}

	uint32_t size_data_compressed(0), size_data_bitpacked(0), size_data_uncompressed(0);
	int seeksuccess = fseek( file(), Header::SIZE_D_UNCOMPRESSED, SEEK_SET );
	if ( !seeksuccess )
	{
		uint16_t data_offset;
		fread( &size_data_uncompressed, Header::field_size.SIZE_D_UNCOMPRESSED, 1, file() );
		fread( &size_data_bitpacked, Header::field_size.SIZE_D_BITPACKED, 1, file() );
		fread( &size_data_compressed, Header::field_size.SIZE_D_COMPRESSED, 1, file() );
	}
	else
	{
		throw std::runtime_error( "fseek to data size details failed: " + std::to_string( seeksuccess ) );
	}

	int seeksuccess = fseek( file(), data_offset, SEEK_SET );
	if ( !seeksuccess )
	{
		ArrayHandle compressed_data(size_data_compressed);
		fread( compressed_data(), sizeof( std::uint8_t ), file() );
	}
	else
	{
		throw std::runtime_error( "fseek to beginning of graph data failed: " + std::to_string( seeksuccess ) );
	}

	return data;
}


Graph::Graph::Graph( const std::string fname, const MemoryLocation memloc ) : memloc( memloc ), get( *this )
{
	FILEIO::FileHandle file( fname, "rb" );
	//Check the file type.
	char filetype[ 8 ];
	fread( filetype, sizeof( char ), 8, file() );
	if ( "xKGRAPHx" == filetype ) {
		std::uint8_t fileversion[ 2 ];
		//Check file format version
		fread( fileversion, sizeof( std::uint8_t ), 2, file() );
		if ( fileversion[ 0 ] <= _MAJOR_VERSION_ && fileversion[ 1 ] <= _MINOR_VERSION_ )
		{
		
			//Once data is loaded into host memory it either kept in host memory or transfered to device memory.
			if ( MemoryLocation::host == memloc )
			{
				number_of_nodes = num_of_nodes;
				adjmat = data.adjmat;
				adjlist = data.adjlist;
				degrees = data.degrees;
				offsets = data.offsets;
			}
			else if ( MemoryLocation::device == memloc )
			{
				cudaMalloc( (void**) &number_of_nodes, sizeof( std::int32_t ) );
				cudaMalloc( (void**) &adjmat, sizeof( std::uint8_t )*number_of_nodes_in_graph*number_of_nodes_in_graph );
				cudaMalloc( (void**) &adjlist, sizeof( std::int32_t )*data.offsets[ number_of_nodes_in_graph + 1 ] );
				cudaMalloc( (void**) &degrees, sizeof( std::int32_t )*number_of_nodes_in_graph );
				cudaMalloc( (void**) &offsets, sizeof( std::uint32_t )*( number_of_nodes_in_graph + 1 ) );

				cudaMemcpy( number_of_nodes, data.num_of_nodes, sizeof( std::int32_t ), cudaMemcpyHostToDevice );
				cudaMemcpy( adjmat, data.adjmat, sizeof( std::uint8_t )*number_of_nodes_in_graph*number_of_nodes_in_graph, cudaMemcpyHostToDevice );
				cudaMemcpy( adjlist, data.adjlist, sizeof( std::int32_t )*data.offsets[ number_of_nodes_in_graph + 1 ], cudaMemcpyHostToDevice );
				cudaMemcpy( degrees, data.degrees, sizeof( int32_t )*number_of_nodes_in_graph, cudaMemcpyHostToDevice );
				cudaMemcpy( offsets, data.offsets, sizeof( std::uint32_t )*( number_of_nodes_in_graph + 1 ), cudaMemcpyHostToDevice );
			}

			data.num_of_nodes = nullptr;
			data.adjmat = nullptr;
			data.adjlist = nullptr;
			data.degrees = nullptr;
			data.offsets = nullptr;
		}
		else
		{
			throw std::runtime_error( "File Format Version: The format version is higher than this library can read. Update library to latest release." );
		}
	}
	else
	{
		throw std::runtime_error( "Incorrect File Format: File is not a .kgraph" );
	}
}

Graph::Graph::~Graph()
{
	if ( MemoryLocation::host == get.memory_location() )
	{
		delete number_of_nodes;
		delete[] adjmat;
		delete[] adjlist;
		delete[] degrees;
		delete[] offsets;
	}
	else if ( MemoryLocation::device == get.memory_location() )
	{
		cudaFree( adjmat );
		cudaFree( adjlist );
		cudaFree( degrees );
		cudaFree( offsets );
	}
}