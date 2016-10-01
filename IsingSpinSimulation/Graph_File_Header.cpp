#include "Graph_FIle_Header.h"

using namespace kspace;
using namespace GRAPH;

template <class T>
T mycast<T>( const std::uint8_t num )
{
	return ( *(T*) &num );
}

void Header::init()
{
	std::uint8_t MajorVersion( _MAJOR_VERSION_ ), MinorVersion( _MINOR_VERSION_ );
	std::uint16_t parameter_offset( 50 );

	memcpy( &header[ VERSION_MAJOR ], &MajorVersion, field_size.VERSION_MAJOR );
	memcpy( &header[ VERSION_MINOR ], &MajorVersion, field_size.VERSION_MINOR );
	memcpy( &header[ OFFSET_PARAMETERS ], &parameter_offset, field_size.OFFSET_PARAMETERS );
}

void Header::clear_parameters()
{
	memset( header + 50, 0, 20 );
}

Header::Header() : get( *this ), set( *this )
{
	init();
}

Header::Header( Parameters &parameters ) : get( *this ), set( *this )
{
	init();
}

Header::Header( const FILEIO::FileHandle& file ) : get( *this ), set( *this )
{

}


std::uint8_t Header::GET::major() const
{
	return *(( std::uint8_t* ) &parent.header[ VERSION_MAJOR ]);
}

std::uint8_t Header::GET::minor() const
{
	return *( ( std::uint8_t* ) &parent.header[ VERSION_MINOR ] );
}

std::uint16_t Header::GET::offset_parameters() const
{
	return *( ( std::uint16_t* ) &parent.header[ OFFSET_PARAMETERS ] );
}

std::uint16_t Header::GET::offset_data() const
{
	return *( ( std::uint16_t* ) &parent.header[ OFFSET_DATA ] );
}

bool Header::GET::is_compressed() const
{
	if ( 0 < size_data_compressed() )
	{
		return true;
	}
	return false;
}

bool Header::GET::is_bitpacked() const
{
	if ( 0 < size_data_bitpacked() )
	{
		return true;
	}
	return false;
}

std::uint32_t Header::GET::size_parameters_uncompressed() const
{
	return *( ( std::uint32_t* ) &parent.header[ SIZE_P_UNCOMPRESSED ] );
}

std::uint32_t Header::GET::size_parameters_compressed() const
{
	return *( ( std::uint32_t* ) &parent.header[ SIZE_P_COMPRESSED ] );
}

std::uint32_t Header::GET::size_data_uncompressed() const
{
	return *( ( std::uint32_t* ) &parent.header[ SIZE_D_UNCOMPRESSED ] );
}

std::uint32_t Header::GET::size_data_bitpacked() const
{
	return *( ( std::uint32_t* ) &parent.header[ SIZE_D_BITPACKED ] );
}

std::uint32_t Header::GET::size_data_compressed() const
{
	return *( ( std::uint32_t* ) &parent.header[ SIZE_D_COMPRESSED ] );
}

ID Header::GET::id() const
{
	return *( ( ID* ) &parent.header[ GRAPH_ID ] );
}

std::int32_t Header::GET::number_of_nodes() const
{
	return *( ( std::int32_t* ) &parent.header[ NUMBER_OF_NODES_IN_GRAPH ] );
}

std::int32_t Header::GET::number_of_nodes_in_file() const
{
	return *( ( std::int32_t* ) &parent.header[ NUMBER_OF_NODES_IN_FILE ] );
}

std::uint32_t Header::GET::number_of_neighbour_in_file() const
{
	return *( ( std::uint32_t* ) &parent.header[ NUMBER_OF_NEIGHBOURS_IN_FILE ] );
}


void Header::SET::major( const std::uint8_t v)
{
	memcpy( &parent.header[ VERSION_MAJOR ], &v, field_size.VERSION_MAJOR );
}

void Header::SET::minor( const std::uint8_t v)
{
	memcpy( &parent.header[ VERSION_MINOR ], &v, field_size.VERSION_MINOR );
}

void Header::SET::offset_data( const std::uint16_t v)
{
	memcpy( &parent.header[ OFFSET_DATA ], &v, field_size.OFFSET_DATA );
}

void Header::SET::size_parameters_uncompressed( const std::uint32_t v)
{
	memcpy( &parent.header[ SIZE_P_UNCOMPRESSED ], &v, field_size.SIZE_P_UNCOMPRESSED );
}

void Header::SET::size_parameters_compressed( const std::uint32_t v)
{
	memcpy( &parent.header[ SIZE_P_COMPRESSED ], &v, field_size.SIZE_P_COMPRESSED );
}

void Header::SET::size_data_uncompressed( const std::uint32_t v)
{
	memcpy( &parent.header[ SIZE_D_UNCOMPRESSED ], &v, field_size.SIZE_D_UNCOMPRESSED );
}

void Header::SET::size_data_bitpacked( const std::uint32_t v)
{
	memcpy( &parent.header[ SIZE_D_BITPACKED ], &v, field_size.SIZE_D_BITPACKED);
}

void Header::SET::size_data_compressed( const std::uint32_t v)
{
	memcpy( &parent.header[ SIZE_D_COMPRESSED ], &v, field_size.SIZE_D_COMPRESSED);
}

void Header::SET::id( const ID v)
{
	memcpy( &parent.header[ GRAPH_ID ], &v, field_size.GRAPH_ID);
}

void Header::SET::number_of_nodes_in_graph( const std::int32_t v)
{
	memcpy( &parent.header[ NUMBER_OF_NODES_IN_GRAPH ], &v, field_size.NUMBER_OF_NODES_IN_GRAPH);
}
void Header::SET::number_of_nodes_in_file( const std::int32_t v)
{
	memcpy( &parent.header[ NUMBER_OF_NODES_IN_FILE ], &v, field_size.NUMBER_OF_NODES_IN_FILE );
}

void Header::SET::number_of_neighbours_in_file( const std::uint32_t v)
{
	memcpy( &parent.header[ NUMBER_OF_NEIGHBOURS_IN_FILE ], &v, field_size.NUMBER_OF_NEIGHBOURS_IN_FILE );

}