#include "Graph_FIle_Header.h"

using namespace kspace;
using namespace Graph;

template <class T>
T mycast( const std::uint8_t num )
{
	return ( *(T*) &num );
}

void Header::init()
{
	std::uint8_t MajorVersion( _MAJOR_VERSION_ ), MinorVersion( _MINOR_VERSION_ );
	std::uint16_t parameter_offset( header_size );

	memcpy( &header[ (size_t) OFFSET::VERSION_MAJOR	],		&MajorVersion,		(size_t) FIELD_SIZE::VERSION_MAJOR );
	memcpy( &header[ (size_t) OFFSET::VERSION_MINOR ],		&MajorVersion,		(size_t) FIELD_SIZE::VERSION_MINOR );
	memcpy( &header[ (size_t) OFFSET::OFFSET_PARAMETERS ],	&parameter_offset,	(size_t) FIELD_SIZE::OFFSET_PARAMETERS );
}

Header::Header() : get( *this ), set( *this )
{
	init();
}

Header::Header( const FILEIO::FileHandle& file ) : get( *this ), set( *this )
{
	if ( nullptr != file() )
	{
		fread( header, sizeof( std::uint8_t ), header_size, file() );
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::uint8_t Header::GET::major() const
{
	return *( ( std::uint8_t* ) &parent.header[ (size_t) OFFSET::VERSION_MAJOR ] );
}

std::uint8_t Header::GET::minor() const
{
	return *( ( std::uint8_t* ) &parent.header[ (size_t) OFFSET::VERSION_MINOR ] );
}

std::uint16_t Header::GET::offset_parameters() const
{
	return *( ( std::uint16_t* ) &parent.header[ (size_t) OFFSET::OFFSET_PARAMETERS ] );
}

std::uint16_t Header::GET::offset_data() const
{
	return *( ( std::uint16_t* ) &parent.header[ (size_t) OFFSET::OFFSET_DATA ] );
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
	return *( ( std::uint32_t* ) &parent.header[ (size_t) OFFSET::SIZE_P_UNCOMPRESSED ] );
}

std::uint32_t Header::GET::size_parameters_compressed() const
{
	return *( ( std::uint32_t* ) &parent.header[ (size_t) OFFSET::SIZE_P_COMPRESSED ] );
}

std::uint32_t Header::GET::size_data_uncompressed() const
{
	return *( ( std::uint32_t* ) &parent.header[ (size_t) OFFSET::SIZE_D_UNCOMPRESSED ] );
}

std::uint32_t Header::GET::size_data_bitpacked() const
{
	return *( ( std::uint32_t* ) &parent.header[ (size_t) OFFSET::SIZE_D_BITPACKED ] );
}

std::uint32_t Header::GET::size_data_compressed() const
{
	return *( ( std::uint32_t* ) &parent.header[ (size_t) OFFSET::SIZE_D_COMPRESSED ] );
}

std::int32_t Header::GET::number_of_nodes_in_graph() const
{
	return *( ( std::int32_t* ) &parent.header[ (size_t) OFFSET::NUMBER_OF_NODES_IN_GRAPH ] );
}

std::uint32_t Header::GET::number_of_edges_in_graph() const
{
	return *( ( std::uint32_t* ) &parent.header[ (size_t) OFFSET::NUMBER_OF_EDGES_IN_GRAPH ] );
}

ID Header::GET::id() const
{
	return *( (ID*) &parent.header[ (size_t) OFFSET::GRAPH_ID ] );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Header::SET::major( const std::uint8_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::VERSION_MAJOR ], &v, (size_t) FIELD_SIZE::VERSION_MAJOR );
}

void Header::SET::minor( const std::uint8_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::VERSION_MINOR ], &v, (size_t) FIELD_SIZE::VERSION_MINOR );
}

void Header::SET::offset_data( const std::uint16_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::OFFSET_DATA ], &v, (size_t) FIELD_SIZE::OFFSET_DATA );
}

void Header::SET::size_parameters_uncompressed( const std::uint32_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::SIZE_P_UNCOMPRESSED ], &v, (size_t) FIELD_SIZE::SIZE_P_UNCOMPRESSED );
}

void Header::SET::size_parameters_compressed( const std::uint32_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::SIZE_P_COMPRESSED ], &v, (size_t) FIELD_SIZE::SIZE_P_COMPRESSED );
}

void Header::SET::size_data_uncompressed( const std::uint32_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::SIZE_D_UNCOMPRESSED ], &v, (size_t) FIELD_SIZE::SIZE_D_UNCOMPRESSED );
}

void Header::SET::size_data_bitpacked( const std::uint32_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::SIZE_D_BITPACKED ], &v, (size_t) FIELD_SIZE::SIZE_D_BITPACKED );
}

void Header::SET::size_data_compressed( const std::uint32_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::SIZE_D_COMPRESSED ], &v, (size_t) FIELD_SIZE::SIZE_D_COMPRESSED );
}

void Header::SET::number_of_nodes_in_graph( const std::int32_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::NUMBER_OF_NODES_IN_GRAPH ], &v, (size_t) FIELD_SIZE::NUMBER_OF_NODES_IN_GRAPH );
}

void Header::SET::number_of_edges_in_graph( const std::uint32_t v )
{
	memcpy( &parent.header[ (size_t) OFFSET::NUMBER_OF_EDGES_IN_GRAPH ], &v, (size_t) FIELD_SIZE::NUMBER_OF_EDGES_IN_GRAPH );
}

void Header::SET::id( const ID v )
{
	memcpy( &parent.header[ (size_t) OFFSET::GRAPH_ID ], &v, (size_t) FIELD_SIZE::GRAPH_ID );
}