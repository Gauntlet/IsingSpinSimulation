#include "Graph.h"
#include <fstream>
#include "File_IO.h"
#include <map>
#include <memory>

using namespace kspace;


struct Data {
	std::int32_t* num_of_nodes;
	std::uint8_t* adjmat;
	std::int32_t* adjlist;
	std::int32_t* degrees;
	std::uint32_t* offsets;
};

Data readData( const FileHandle &file, const uint16_t data_offset, const uint32_t data_size, const Graph::GraphType graphtype, const uint32_t number_of_nodes_in_graph, const uint32_t number_of_nodes_in_file, const uint32_t number_of_neighbours )
{
	Data data = { nullptr, nullptr, nullptr, nullptr, nullptr };

	int seeksuccess = fseek( file(), data_offset, SEEK_SET );
	if ( !seeksuccess )
	{
		data.num_of_nodes = new std::int32_t( number_of_nodes_in_graph );
		//create arrays to store graph data.
		data.adjmat = new std::uint8_t[ number_of_nodes_in_graph * number_of_nodes_in_graph ]();
		data.degrees = new std::int32_t[ number_of_nodes_in_graph ]();
		data.offsets = new std::uint32_t[ number_of_nodes_in_graph + 1 ]();

		fread( data.degrees, sizeof( uint32_t ), number_of_nodes_in_graph, file() );
		for ( size_t i = 0; i < number_of_nodes_in_graph; ++i )
		{
			data.offsets[ i + 1 ] = data.offsets[ i ] + data.degrees[ i ];
		}
		data.adjlist = new std::int32_t[ number_of_nodes_in_graph ]();

		//create arrays to store graph file data.
		std::unique_ptr<std::int32_t[]> vertex_ids( new int32_t[ number_of_nodes_in_file ]() );
		std::unique_ptr<std::uint32_t[]> neighbour_offsets( new uint32_t[ number_of_nodes_in_file + 1 ]() );
		std::unique_ptr<std::int32_t[]> neighbour_ids( new int32_t[ number_of_neighbours ]() );
		std::unique_ptr<std::uint32_t[]> degree_counter( new uint32_t[ number_of_nodes_in_graph ]() );

		fread( vertex_ids.get(), sizeof( int32_t ), number_of_nodes_in_file, file() );
		fread( neighbour_offsets.get(), sizeof( uint32_t ), number_of_nodes_in_file + 1, file() );
		fread( neighbour_ids.get(), sizeof( int32_t ), number_of_neighbours, file() );

		size_t v, w;

		//Transform graph file data into graph data.
		for ( size_t i = 0; i < number_of_nodes_in_file; ++i )
		{
			v = vertex_ids[ i ] - 1;
			for ( size_t j = neighbour_offsets[ i ]; j < neighbour_offsets[ i + 1 ]; ++j )
			{
				w = abs( neighbour_ids[ j ] ) - 1;
				size_t index = v * number_of_neighbours + w;
				size_t offset = data.offsets[ v ];

				data.adjmat[ index ] = 1;
				data.adjlist[ offset + degree_counter[ v ] ] = w;
				degree_counter[ v ]++;

				if ( neighbour_ids[ j ] >= 0 )
				{
					index = w * number_of_neighbours + v;
					offset = data.offsets[ w ];

					data.adjmat[ index ] = 1;
					data.adjlist[ offset + degree_counter[ w ] ] = v;
					degree_counter[ w ]++;
				}
			}
		}
	}
	else
	{
		throw std::runtime_error( "fseek to beginning of graph data failed: " + std::to_string( seeksuccess ) );
	}

	return data;
}


Graph::Graph::Graph( const std::string fname, const MemoryLocation memloc )
{
	FileHandle file( fname, "rb" );
	//Check the file type.
	char filetype[ 8 ];
	fread( filetype, sizeof( char ), 8, file() );
	if ( "xKGRAPHx" == filetype ) {
		std::uint8_t fileversion[ 2 ];
		//Check file format version
		fread( fileversion, sizeof( std::uint8_t ), 2, file() );
		if ( fileversion[ 0 ] <= MAJOR_VERSION && fileversion[ 1 ] <= MINOR_VERSION )
		{
			std::uint16_t parameter_offset, data_offset, graphtype;
			std::uint32_t data_size, number_of_neighbours;
			std::int32_t number_of_nodes_in_graph, number_of_nodes_in_file;

			fread( &parameter_offset, sizeof( std::uint16_t ), 1, file() );			//parameter offset
			fread( &data_offset, sizeof( std::uint16_t ), 1, file() );				//Data offset
			fread( &data_size, sizeof( std::uint32_t ), 1, file() );				//Data size (B)
			fread( &graphtype, sizeof( uint16_t ), 1, file() );						//Graph type
			fread( &number_of_nodes_in_graph, sizeof( std::int32_t ), 1, file() );	//Number of nodes in the graph
			fread( &number_of_nodes_in_file, sizeof( std::int32_t ), 1, file() );	//Number of nodes in the file
			fread( &number_of_neighbours, sizeof( std::uint32_t ), 1, file() );		//Number of neighbours in the file

			Data data = readData( file, data_offset, data_size, (GraphType) graphtype, number_of_nodes_in_graph, number_of_nodes_in_file, number_of_neighbours );

			_memloc = memloc;
			//Once data is loaded into host memory it either kept in host memory or transfered to device memory.
			if ( MemoryLocation::host == memloc )
			{
				_number_of_nodes = data.num_of_nodes;
				_adjmat = data.adjmat;
				_adjlist = data.adjlist;
				_degrees = data.degrees;
				_offsets = data.offsets;
			}
			else if ( MemoryLocation::device == memloc )
			{
				cudaMalloc( (void**) &_number_of_nodes, sizeof( std::int32_t ) );
				cudaMalloc( (void**) &_adjmat, sizeof( std::uint8_t )*number_of_nodes_in_graph*number_of_nodes_in_graph );
				cudaMalloc( (void**) &_adjlist, sizeof( std::int32_t )*data.offsets[ number_of_nodes_in_graph + 1 ] );
				cudaMalloc( (void**) &_degrees, sizeof( std::int32_t )*number_of_nodes_in_graph );
				cudaMalloc( (void**) &_offsets, sizeof( std::uint32_t )*( number_of_nodes_in_graph + 1 ) );

				cudaMemcpy( _number_of_nodes, data.num_of_nodes, sizeof( std::int32_t ), cudaMemcpyHostToDevice );
				cudaMemcpy( _adjmat, data.adjmat, sizeof( std::uint8_t )*number_of_nodes_in_graph*number_of_nodes_in_graph, cudaMemcpyHostToDevice );
				cudaMemcpy( _adjlist, data.adjlist, sizeof( std::int32_t )*data.offsets[ number_of_nodes_in_graph + 1 ], cudaMemcpyHostToDevice );
				cudaMemcpy( _degrees, data.degrees, sizeof( int32_t )*number_of_nodes_in_graph, cudaMemcpyHostToDevice );
				cudaMemcpy( _offsets, data.offsets, sizeof( std::uint32_t )*( number_of_nodes_in_graph + 1 ), cudaMemcpyHostToDevice );
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
	if ( MemoryLocation::host == memory_location() )
	{
		delete _number_of_nodes;
		delete[] _adjmat;
		delete[] _adjlist;
		delete[] _degrees;
		delete[] _offsets;
	}
	else if ( MemoryLocation::device == memory_location() )
	{
		cudaFree( _adjmat );
		cudaFree( _adjlist );
		cudaFree( _degrees );
		cudaFree( _offsets );
	}
}