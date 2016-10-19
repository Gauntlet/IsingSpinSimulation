
#include "File_IO.h"
#include "ArrayHandle.h"
#include "Compression.h"
#include <vector>

#include "details.h"
#include "Graph.h"
#include "Graph_File_Header.h"

using namespace kspace::GRAPH;
using namespace kspace::FILEIO;

Graph::Data::~Data()
{
	if ( MemoryLocation::host == memloc )
	{
		delete number_of_nodes;
		delete[] adjmat;
		delete[] adjlist;
		delete[] degrees;
		delete[] offsets;
	}
	else if ( MemoryLocation::device == memloc )
	{
		cudaFree( adjmat );
		cudaFree( adjlist );
		cudaFree( degrees );
		cudaFree( offsets );
	}
}

Graph::Data::Data( Data&& that ) : number_of_nodes( std::move( that.number_of_nodes ) ), adjmat( std::move( that.adjmat ) ), adjlist( std::move( that.adjlist ) ), degrees( std::move( that.degrees ) ), offsets( std::move( that.offsets ) )
{
	that.clear();
}

Graph::Data& Graph::Data::operator=( Data&& that )
{
	number_of_nodes = std::move( that.number_of_nodes );
	adjmat = std::move( that.adjmat );
	adjlist = std::move( that.adjlist );
	degrees = std::move( that.degrees );
	offsets = std::move( that.offsets );

	that.clear();

	return *this;
}

void Graph::Data::clear()
{
	number_of_nodes = nullptr;
	adjmat = nullptr;
	adjlist = nullptr;
	degrees = nullptr;
	offsets = nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Graph::Data Graph::readData( const FILEIO::FileHandle &file )
{
	Data tmpdata;

	Header header( file );

	uint32_t size_data_compressed( 0 ), size_data_bitpacked( 0 ), size_data_uncompressed( 0 );
	int seeksuccess = fseek( file(), header.get.offset_data(), SEEK_SET );

	if ( !seeksuccess )
	{
		ArrayHandle<Byte> compressed_data( header.get.size_data_compressed() );
		fread( compressed_data.set.data_ptr(), sizeof( std::uint8_t ), header.get.size_data_compressed(), file() );

		Compression compression;

		Compression::Details det;
		det.compressed_data_size = header.get.size_data_compressed();
		det.bitpacked_data_size = header.get.size_data_bitpacked();
		det.uncompressed_data_size = header.get.size_data_uncompressed();
		if ( det.bitpacked_data_size > 0 )
		{
			det.bits.zerobit = 0;
			det.bits.unitbit = 1;
		}

		ArrayHandle<Byte> uncompressed_data = std::move( compression.inflate( compressed_data, det ) );

		const std::int32_t N = header.get.number_of_nodes_in_graph();

		tmpdata.number_of_nodes = new std::int32_t( N );
		tmpdata.adjmat = new std::uint8_t[ N*N ]();
		tmpdata.degrees = new std::int32_t[ N ]();
		tmpdata.offsets = new std::uint32_t[ N + 1 ]();
		tmpdata.adjlist = new std::int32_t[ header.get.number_of_edges_in_graph() ];

		//Copy the adjacency matrix to data structure.
		memcpy( tmpdata.number_of_nodes, uncompressed_data.get.data_ptr(), sizeof( std::uint8_t )*N*N );

		//Create the degree, offset and adjacency list from the adjacency matrix.
		for ( size_t i = 0; i < N; ++i )
		{
			for ( size_t j = 0; j < N; ++j )
			{
				if ( data.adjmat[ i*N + j ] > 0 )
				{
					tmpdata.adjlist[ tmpdata.offsets[ i ] + tmpdata.degrees[ i ] ] = j;
					tmpdata.degrees[ i ]++;
				}

			}

			tmpdata.offsets[ i + 1 ] = tmpdata.offsets[ i ] + tmpdata.degrees[ i ];
		}
	}
	else
	{
		throw std::runtime_error( "fseek to beginning of graph data failed: " + std::to_string( seeksuccess ) );
	}

	return tmpdata;
}


Graph::Graph::Graph( const std::string fname, const MemoryLocation memloc ) : get( *this ), set( *this )
{
	FILEIO::FileHandle file( fname, "rb" );
	//Check the file type.
	char filetype[ 8 ];
	fread( filetype, sizeof( char ), 8, file() );
	if ( "xKGRAPHx" == filetype )
	{
		std::uint8_t fileversion[ 2 ];
		//Check file format version
		fread( fileversion, sizeof( std::uint8_t ), 2, file() );
		if ( fileversion[ 0 ] <= _MAJOR_VERSION_ && fileversion[ 1 ] <= _MINOR_VERSION_ )
		{

			Data tmpdata = readData( file );
			const std::uint32_t N = *data.number_of_nodes;
			//Once data is loaded into host memory it either kept in host memory or transfered to device memory.
			if ( MemoryLocation::host == memloc )
			{
				data = std::move( tmpdata );
			}
			else if ( MemoryLocation::device == memloc )
			{
				data.memloc = MemoryLocation::device;

				HANDLE_ERROR( cudaMalloc( (void**) &data.number_of_nodes, sizeof( std::int32_t ) ) );
				HANDLE_ERROR( cudaMalloc( (void**) &data.adjmat, sizeof( std::uint8_t )*N*N ) );
				HANDLE_ERROR( cudaMalloc( (void**) &data.adjlist, sizeof( std::int32_t )*tmpdata.offsets[ N + 1 ] ) );
				HANDLE_ERROR( cudaMalloc( (void**) &data.degrees, sizeof( std::int32_t )*N ) );
				HANDLE_ERROR( cudaMalloc( (void**) &data.offsets, sizeof( std::uint32_t )*( N + 1 ) ) );

				HANDLE_ERROR( cudaMemcpy( data.number_of_nodes, &data.number_of_nodes, sizeof( std::int32_t ), cudaMemcpyHostToDevice ) );
				HANDLE_ERROR( cudaMemcpy( data.adjmat, data.adjmat, sizeof( std::uint8_t )*N*N, cudaMemcpyHostToDevice ) );
				HANDLE_ERROR( cudaMemcpy( data.adjlist, data.adjlist, sizeof( std::int32_t )*tmpdata.offsets[ N + 1 ], cudaMemcpyHostToDevice ) );
				HANDLE_ERROR( cudaMemcpy( data.degrees, data.degrees, sizeof( int32_t )*N, cudaMemcpyHostToDevice ) );
				HANDLE_ERROR( cudaMemcpy( data.offsets, data.offsets, sizeof( std::uint32_t )*( N + 1 ), cudaMemcpyHostToDevice ) );
			}
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t const & Graph::GET::number_of_nodes() const
{
	return *parent.data.number_of_nodes;
}

int32_t const & Graph::GET::degree( const uint32_t v ) const
{
	if ( number_of_nodes() <= v )
	{
		throw std::out_of_range( "degree() index is out of bounds: v=" + std::to_string(v) );
	}
	return parent.data.degrees[ v ];
}

std::uint32_t const & Graph::GET::offset( const size_t v ) const
{
	if ( number_of_nodes() <= v )
	{
		throw std::out_of_range( "offset() index is out of bounds: v=" + std::to_string( v ) );
	}

	return parent.data.offsets[ v ];
}

bool const & Graph::GET::is_connected( const uint32_t v, const uint32_t w ) const
{
	if ( number_of_nodes() <= v || number_of_nodes() <= w )
	{
		throw std::out_of_range( "is_connected() indices are out of bounds: (v,w)=(" + std::to_string( v ) + "," + std::to_string(w) + ")" );
	}

	return parent.data.adjmat[ v * number_of_nodes() + w ];
}

int32_t const & Graph::GET::neighbour( const uint32_t v, const uint32_t kth_neighbour ) const
{
	if ( number_of_nodes() <= v || number_of_nodes() <= kth_neighbour )
	{
		throw std::out_of_range( "neighbour() indices are out of bounds: (v,kth_neighbour)=(" + std::to_string( v ) + "," + std::to_string( kth_neighbour ) + ")" );
	}

	return parent.data.adjlist[ offset( v ) + kth_neighbour ];
}

kspace::MemoryLocation const & Graph::GET::memory_location() const
{
	return parent.data.memloc;
}

std::uint8_t const * Graph::GET::adjmat() const
{
	return parent.data.adjmat;
}

std::int32_t const * Graph::GET::adjlist() const
{
	return parent.data.adjlist;
}

std::int32_t const * Graph::GET::degrees() const
{
	return parent.data.degrees;
}

std::uint32_t const * Graph::GET::offsets() const
{
	return parent.data.offsets;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::uint8_t* Graph::SET::adjmat() const
{
	return parent.data.adjmat;
}

std::int32_t* Graph::SET::adjlist() const
{
	return parent.data.adjlist;
}

std::int32_t* Graph::SET::degrees() const
{
	return parent.data.degrees;
}

std::uint32_t* Graph::SET::offsets() const
{
	return parent.data.offsets;
}