
#include "File_IO.h"
#include "Compression.h"
#include "Array.h"
#include "details.h"
#include "Graph.h"
#include "Graph_File_Header.h"

#include <vector>

using namespace kspace::GRAPH;
using namespace kspace::FILEIO;

void Graph::Data::move_data( Data&& that )
{
	clear();

	memloc = that.memloc;
	number_of_nodes = that.number_of_nodes;
	adjmat = that.adjmat;
	adjlist = that.adjlist;
	degrees = that.degrees;
	offsets = that.offsets;

	that.memloc = MemoryLocation::host;
	that.number_of_nodes = nullptr;
	that.adjmat = nullptr;
	that.adjlist = nullptr;
	that.degrees = nullptr;
	that.offsets = nullptr;
}

/**
* Frees the resources being managed. 
*/
Graph::Data::~Data()
{
	clear();
}

/**
* Moves the resources managed by the passed Data object to the one being constructed.
* @param that.
*/
Graph::Data::Data( Data&& that )
{
	move_data( std::move(that) );
}

/**
* Moves the resources managed by the Data object on the RHS to the one on the LHS.
* @param that.
*/
Graph::Data& Graph::Data::operator=( Data&& that )
{
	move_data( std::move(that) );

	return *this;
}

/**
* Frees the resources being managed by the data object.
*/
void Graph::Data::clear()
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
		HANDLE_ERROR(cudaFree( number_of_nodes ));
		HANDLE_ERROR( cudaFree( adjmat ) );
		HANDLE_ERROR( cudaFree( adjlist ) );
		HANDLE_ERROR( cudaFree( degrees ) );
		HANDLE_ERROR( cudaFree( offsets ) );
	}

	memloc = MemoryLocation::host;
	number_of_nodes = nullptr;
	adjmat = nullptr;
	adjlist = nullptr;
	degrees = nullptr;
	offsets = nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* A helper function which reads the graph data stored in a file and reconstructs it into a useable format.
*
* @param file a FileHandle to a kgraph file.
* @return
*/
Graph::Data&& Graph::readData( const FILEIO::FileHandle &file )
{
	Data tmpdata;

	Header header( file );

	uint32_t size_data_compressed( 0 ), size_data_bitpacked( 0 ), size_data_uncompressed( 0 );
	int seeksuccess = fseek( file(), header.get.offset_data(), SEEK_SET );

	if ( !seeksuccess )
	{
		Array<Byte> compressed_data( header.get.size_data_compressed() );
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

		Array<Byte> uncompressed_data = std::move( compression.inflate( compressed_data, det ) );

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

	return std::move(tmpdata);
}

/**
* A helper function for initialising the graph from a data object.
*/
void Graph::Graph::initialise( Data& tmpdata, MemoryLocation const & memloc )
{
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

		HANDLE_ERROR( cudaMemcpy( data.number_of_nodes, &tmpdata.number_of_nodes, sizeof( std::int32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( data.adjmat, tmpdata.adjmat, sizeof( std::uint8_t )*N*N, cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( data.adjlist, tmpdata.adjlist, sizeof( std::int32_t )*tmpdata.offsets[ N + 1 ], cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( data.degrees, tmpdata.degrees, sizeof( int32_t )*N, cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( data.offsets, tmpdata.offsets, sizeof( std::uint32_t )*( N + 1 ), cudaMemcpyHostToDevice ) );
	}
}

/**
* On construction a Graph loads data from a file given by 'fname' and stored in the device or host memory as indicated.
* @param fname a std::string containing the full path to a kgraph file.
* @param memloc a enum indicating whether the graph data should be accessible by device or host processes.
*/
Graph::Graph( std::string const & fname, MemoryLocation const & memloc ) : get( *this ), set( *this )
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
			initialise( tmpdata, memloc );
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

/**
* When generating graphs we intially store the data in a Graph::Data object and pass that into
* a Graph object to be properly managed.
* @param data a Graph::Data object containing pointers to the adjacency matrix, adjacency list, list offsets and degrees.
* @param memloc a enum indicating whether the graph data should be accessible by device or host processes.
*/
Graph::Graph(Data& data, MemoryLocation const & memloc) : get(*this), set(*this)
{
	initialise( data, memloc );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* The number of nodes in the graph.
*
* Read only access.
*/
int32_t const & Graph::GRAPH_GET::number_of_nodes() const
{
	return *parent.data.number_of_nodes;
}

/**
* The number of edges attached to the specified node.
*
* Read only access.
* @param v an uint32_t node id.
* @return an signed integer.
*/
int32_t const & Graph::GRAPH_GET::degree( const uint32_t v ) const
{
	if ( number_of_nodes() <= v )
	{
		throw std::out_of_range( "degree() index is out of bounds: v=" + std::to_string( v ) );
	}
	return parent.data.degrees[ v ];
}

/**
* The index in the adjacency list where the list of neighbours for the specified node begins.
*
* Read only access.
* @param v an uint32_t node id. 
* @return a uint32_t.
*/
std::uint32_t const & Graph::GRAPH_GET::offset( const size_t v ) const
{
	if ( number_of_nodes() <= v )
	{
		throw std::out_of_range( "offset() index is out of bounds: v=" + std::to_string( v ) );
	}

	return parent.data.offsets[ v ];
}

/**
* Returns whether two vertices are connected by an edge.
*
* Read only access.
* @param v an uint32_t node id.
* @param w an uint32_t node id.
* @return a bool.
*/
bool const & Graph::GRAPH_GET::is_connected( const uint32_t v, const uint32_t w ) const
{
	if ( number_of_nodes() <= v || number_of_nodes() <= w )
	{
		throw std::out_of_range( "is_connected() indices are out of bounds: (v,w)=(" + std::to_string( v ) + "," + std::to_string( w ) + ")" );
	}

	return parent.data.adjmat[ v * number_of_nodes() + w ];
}

/**
* Returns the node id of k-th neighbour of the specified node.
*
* Read only access.
* @param v an uint32_t node id.
* @param kth_neighbour an uint32_t list index.
* @param an int32_t.
*/
int32_t const & Graph::GRAPH_GET::neighbour( const uint32_t v, const uint32_t kth_neighbour ) const
{
	if ( number_of_nodes() <= v || number_of_nodes() <= kth_neighbour )
	{
		throw std::out_of_range( "neighbour() indices are out of bounds: (v,kth_neighbour)=(" + std::to_string( v ) + "," + std::to_string( kth_neighbour ) + ")" );
	}

	return parent.data.adjlist[ offset( v ) + kth_neighbour ];
}

/**
* Returns the memory location of the resources being managed.
*
* Read only access.
* @return an enum.
*/
kspace::MemoryLocation const & Graph::GRAPH_GET::memory_location() const
{
	return parent.data.memloc;
}

/**
* Returns the pointer to the adjacency matrix which is formatted as a column ordered 1D array.
*
* Read only access.
* @return a pointer to an 8bit unsigned integer array.
*/
std::uint8_t const * Graph::GRAPH_GET::adjmat() const
{
	return parent.data.adjmat;
}

/**
* Returns the pointer to the adjacency list which is formatted as a 1D array.
* The beginning position of a node n can be found in the offset list.
*
* Read only access.
* @return a pointer to an 32bit signed integer array.
*/
std::int32_t const * Graph::GRAPH_GET::adjlist() const
{
	return parent.data.adjlist;
}

/**
* Returns the pointer to the degrees list which is formatted as a 1D array.
* The n-th element is the number of edges that are attached to the node n.
*
* Read only access.
* @return a pointer to an 32bit signed integer array.
*/
std::int32_t const * Graph::GRAPH_GET::degrees() const
{
	return parent.data.degrees;
}

/**
* Returns the pointer to the list of offsets which is formatted as a 1D array.
* The offset array contains the beginning position of each list of neighbours.
* Thus the n-th element is an index (in the adjacnecy list array) of the first 
* neighbour of node n.
*
* Read only access.
* @return a pointer to an 32bit signed integer array.
*/
std::uint32_t const * Graph::GRAPH_GET::offsets() const
{
	return parent.data.offsets;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* Returns the pointer to the adjacency matrix which is formatted as a column ordered 1D array.
*
* Read and write access.
* @return a pointer to an 8bit unsigned integer array.
*/
std::uint8_t* Graph::GRAPH_SET::adjmat() const
{
	return parent.data.adjmat;
}

/**
* Returns the pointer to the adjacency list which is formatted as a 1D array.
* The beginning position of a node n can be found in the offset list.
*
* Read and write access.
* @return a pointer to an 32bit signed integer array.
*/
std::int32_t* Graph::GRAPH_SET::adjlist() const
{
	return parent.data.adjlist;
}

/**
* Returns the pointer to the degrees list which is formatted as a 1D array.
* The n-th element is the number of edges that are attached to the node n.
*
* Read and write access.
* @return a pointer to an 32bit signed integer array.
*/
std::int32_t* Graph::GRAPH_SET::degrees() const
{
	return parent.data.degrees;
}

/**
* Returns the pointer to the list of offsets which is formatted as a 1D array.
* The offset array contains the beginning position of each list of neighbours.
* Thus the n-th element is an index (in the adjacnecy list array) of the first
* neighbour of node n.
*
* Read and write access.
* @return a pointer to an 32bit signed integer array.
*/
std::uint32_t* Graph::GRAPH_SET::offsets() const
{
	return parent.data.offsets;
}