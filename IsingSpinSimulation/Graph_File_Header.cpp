#include "Graph_FIle_Header.h"

using namespace kspace;
using namespace GRAPH;

template <class T>
T mycast( const std::uint8_t num )
{
	return ( *(T*) &num );
}

void Header::init()
{
	std::uint8_t MajorVersion( _MAJOR_VERSION_ ), MinorVersion( _MINOR_VERSION_ );
	std::uint16_t parameter_offset( header_size );

	memcpy( &header.set.data_ptr()[ (size_t) OFFSET::VERSION_MAJOR	],		&MajorVersion,		(size_t) FIELD_SIZE::VERSION_MAJOR );
	memcpy( &header.set.data_ptr()[ (size_t) OFFSET::VERSION_MINOR ], &MajorVersion, (size_t) FIELD_SIZE::VERSION_MINOR );
	memcpy( &header.set.data_ptr()[ (size_t) OFFSET::OFFSET_PARAMETERS ], &parameter_offset, (size_t) FIELD_SIZE::OFFSET_PARAMETERS );
}

Header::Header() : header(header_size), get( *this ), set( *this )
{
	init();
}

/**
* Read the header of the passed file.
*/
Header::Header( const FILEIO::FileHandle& file ) : get( *this ), set( *this )
{
	if ( nullptr != file() )
	{
		fread( header.set.data_ptr(), sizeof( std::uint8_t ), header_size, file() );
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* The file format major version.
*/
std::uint8_t Header::HEADER_GET::major() const
{
	return *( ( std::uint8_t* ) &(parent.header.get.data_ptr()[ (size_t) OFFSET::VERSION_MAJOR ] ));
}

/**
* The file format minor version.
*/
std::uint8_t Header::HEADER_GET::minor() const
{
	return *( ( std::uint8_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::VERSION_MINOR ] );
}


/**
* The offset in bytes to the beginning of the graph parameters. 
*/
std::uint16_t Header::HEADER_GET::offset_parameters() const
{
	return *( ( std::uint16_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::OFFSET_PARAMETERS ] );
}

/**
* The offset in bytes to the beginning of the graph data.
*/
std::uint16_t Header::HEADER_GET::offset_data() const
{
	return *( ( std::uint16_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::OFFSET_DATA ] );
}

/**
* Returns whether the graph parameters are compressed or not.
*/
bool Header::HEADER_GET::is_parameters_compressed() const
{
	if ( 0 < size_parameters_compressed() )
	{
		return true;
	}
	return false;
}

/**
* Returns whether the graph data is compressed or not.
*/
bool Header::HEADER_GET::is_data_compressed() const
{
	if ( 0 < size_data_compressed() )
	{
		return true;
	}
	return false;
}

/**
* Returns whether the graph data is bitpacked or not.
*/
bool Header::HEADER_GET::is_data_bitpacked() const
{
	if ( 0 < size_data_bitpacked() )
	{
		return true;
	}
	return false;
}

/**
* The size of the parameter data in its uncompressed state. 
*/
std::uint32_t Header::HEADER_GET::size_parameters_uncompressed() const
{
	return *( ( std::uint32_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::SIZE_P_UNCOMPRESSED ] );
}

/** 
* The size of the parameter data in its compressed state.
*/
std::uint32_t Header::HEADER_GET::size_parameters_compressed() const
{
	return *( ( std::uint32_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::SIZE_P_COMPRESSED ] );
}

/**
* The size of the graph data in its uncompressed state. 
*/
std::uint32_t Header::HEADER_GET::size_data_uncompressed() const
{
	return *( ( std::uint32_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::SIZE_D_UNCOMPRESSED ] );
}

/**
* The size of the graph data in its bitpacked state.
*/
std::uint32_t Header::HEADER_GET::size_data_bitpacked() const
{
	return *( ( std::uint32_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::SIZE_D_BITPACKED ] );
}

/**
* The size of the graph data in its compressed state.
*/
std::uint32_t Header::HEADER_GET::size_data_compressed() const
{
	return *( ( std::uint32_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::SIZE_D_COMPRESSED ] );
}

/**
* The number of nodes in the graph.
*/
std::int32_t Header::HEADER_GET::number_of_nodes_in_graph() const
{
	return *( ( std::int32_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::NUMBER_OF_NODES_IN_GRAPH ] );
}

/**
* The number of edges in the graph.
*/
std::uint32_t Header::HEADER_GET::number_of_edges_in_graph() const
{
	return *( ( std::uint32_t* ) &parent.header.get.data_ptr()[ (size_t) OFFSET::NUMBER_OF_EDGES_IN_GRAPH ] );
}

/**
* The graph ID, indicates what type of graph it is.
*/
Graph::ID Header::HEADER_GET::id() const
{
	return *( ( Graph::ID* ) &parent.header.set.data_ptr()[ (size_t) OFFSET::GRAPH_ID ] );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* Sets the file format major version number.
* @param major_version an uint8_t.
*/
void Header::HEADER_SET::major( const std::uint8_t major_version )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::VERSION_MAJOR ], &major_version, (size_t) FIELD_SIZE::VERSION_MAJOR );
}

/**
* Sets the file format minor version number.
* @param minor_version an uint8_t.
*/
void Header::HEADER_SET::minor( const std::uint8_t minor_version )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::VERSION_MINOR ], &minor_version, (size_t) FIELD_SIZE::VERSION_MINOR );
}

/**
* Sets the number of bytes from the beginning of the file to where the graph data begins.
* @param data_offset an uint32_t.
*/
void Header::HEADER_SET::offset_data( const std::uint16_t data_offset )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::OFFSET_DATA ], &data_offset, (size_t) FIELD_SIZE::OFFSET_DATA );
}

/**
* Sets the size of the parameter data in its uncompressed state. 
* @param uncompressed_parameters_size an uint32_t.
*/
void Header::HEADER_SET::size_parameters_uncompressed( const std::uint32_t uncompressed_parameters_size )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::SIZE_P_UNCOMPRESSED ], &uncompressed_parameters_size, (size_t) FIELD_SIZE::SIZE_P_UNCOMPRESSED );
}

/**
* Sets the size of the parameter data in its compressed state. 
* @param compressed_parameters_size an uint32_t.
*/
void Header::HEADER_SET::size_parameters_compressed( const std::uint32_t compressed_parameters_size )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::SIZE_P_COMPRESSED ], &compressed_parameters_size, (size_t) FIELD_SIZE::SIZE_P_COMPRESSED );
}

/**
* Sets the size of the graph data in its uncompressed state. 
* @param uncompressed_data_size an uint32_t.
*/
void Header::HEADER_SET::size_data_uncompressed( const std::uint32_t uncompressed_data_size )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::SIZE_D_UNCOMPRESSED ], &uncompressed_data_size, (size_t) FIELD_SIZE::SIZE_D_UNCOMPRESSED );
}

/**
* Sets the size of the graph data in its bitpacked state. 
* @param bitpacked_data_size an uint32_t.
*/
void Header::HEADER_SET::size_data_bitpacked( const std::uint32_t bitpacked_data_size )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::SIZE_D_BITPACKED ], &bitpacked_data_size, (size_t) FIELD_SIZE::SIZE_D_BITPACKED );
}

/**
* Sets the size of the graph data in its compressed state.
* @param compressed_data_size an uint32_t.
*/
void Header::HEADER_SET::size_data_compressed( const std::uint32_t compressed_data_size )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::SIZE_D_COMPRESSED ], &compressed_data_size, (size_t) FIELD_SIZE::SIZE_D_COMPRESSED );
}

/**
* Sets the value of the number of nodes in the graph.
* @param number_of_nodes an uint32_t.
*/
void Header::HEADER_SET::number_of_nodes_in_graph( const std::int32_t number_of_nodes )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::NUMBER_OF_NODES_IN_GRAPH ], &number_of_nodes, (size_t) FIELD_SIZE::NUMBER_OF_NODES_IN_GRAPH );
}

/**
* Sets the value of the number of edges in the graph.
* @param number_of_edges an uint32_t.
*/
void Header::HEADER_SET::number_of_edges_in_graph( const std::uint32_t number_of_edges )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::NUMBER_OF_EDGES_IN_GRAPH ], &number_of_edges, (size_t) FIELD_SIZE::NUMBER_OF_EDGES_IN_GRAPH );
}

/**
* Writes the type of graph to the header.
* @param id an enum which indicates the graph type.
*/
void Header::HEADER_SET::id( const Graph::ID id )
{
	memcpy( &parent.header.set.data_ptr()[ (size_t) OFFSET::GRAPH_ID ], &id, (size_t) FIELD_SIZE::GRAPH_ID );
}